#!/bin/env python

import argparse
from pathlib import Path

# from typing import Dict, List, Union
import Bio
import models
import numpy as np
import pyBigWig as pbw
import torch
import train_pytorch
from Bio import SeqIO

# from torch import nn
from torch.utils.data import DataLoader, Dataset


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_file",
        help="pytorch file containing model weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fasta_file",
        help="Fasta file containing genome to predict on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-arch",
        "--architecture",
        help="Name of the model architecture",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-nt",
        "--n_tracks",
        help="Number of tracks predicted by the model (default: %(default)s)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-c",
        "--chroms",
        help='Chromosomes to predict on. Specify "all" for all chromosomes (default: %(default)s)',
        nargs="+",
        type=str,
        default="all",
    )
    parser.add_argument(
        "-s",
        "--strand",
        help='Strand to perform training on, choose between "for", "rev", '
        '"both" or "merge". Merge will average predictions from both strands '
        "to make a single prediction (default: %(default)s)",
        choices=["for", "rev", "both", "merge"],
        default="both",
    )
    parser.add_argument(
        "-w",
        "--winsize",
        help="Size of the window in bp to use for prediction (default: %(default)s)",
        default=2048,
        type=int,
    )
    parser.add_argument(
        "-h_int",
        "--head_interval",
        help="Spacing between output head in case of mutliple outputs "
        "(default: %(default)s)",
        default=16,
        type=int,
    )
    parser.add_argument(
        "-crop",
        "--head_crop",
        help="Number of heads to crop on left and right side of window (default: %(default)s)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-mid",
        "--middle",
        help="Indicates to use only middle heads for prediction",
        action="store_true",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of samples to use for training in parallel (default: %(default)s)",
        default=16384,
        type=int,
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        help="Number of parallel workers for preprocessing batches (default: %(default)s)",
        default=4,
        type=int,
    )
    # parser.add_argument(
    #     "--seed",
    #     help="Seed to use for random shuffling of training samples",
    #     default=None,
    #     type=int)
    parser.add_argument(
        "-v",
        "--verbose",
        help="Indicates to show information messages",
        action="store_true",
    )
    args = parser.parse_args()
    # Check that files exist
    for argname in ["fasta_file", "model_file"]:
        filename = vars(args)[argname]
        if not Path(filename).is_file():
            raise ValueError(
                f"File {filename} does not exist. Please enter a valid {argname}"
            )
    # Check that arguments are valid
    for argname in ["winsize", "head_interval", "batch_size", "num_workers"]:
        arg = vars(args)[argname]
        if arg < 1:
            raise ValueError(f"Argument {argname} ({arg}) must be greater than 0")
    # Format chromosome names
    if args.chroms != "all":
        genome_name = Path(args.fasta_file).stem
        if genome_name == "W303":
            args.chroms = ["chr" + format(int(c), "02d") for c in args.chroms]
        elif genome_name == "mm10":
            args.chroms = [f"chr{c}" for c in args.chroms]
        elif genome_name == "W303_Mmmyco":
            args.chroms = [f"chr{c}" if c != "Mmmyco" else c for c in args.chroms]
    return args


def apply_on_index(func, *args, length=None, neutral=0):
    """Applies a function on arrays specified as indices and values.

    Parameters
    ----------
    func : callable
        Function to compute on arrays, must take the number of values as first
        argument, then as many arrays as provided in args.
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, default=None
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.
    neutral : int, default=0
        Neutral element for the desired operation. This function requires that
        func has a neutral element.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    idxes, vals = zip(*args)
    if length is None:
        length = np.max([np.max(idx) for idx in idxes]) + 1
    n_per_pos = np.zeros((1, length))
    newvals = []
    for idx, val in args:
        val2D = val.reshape(-1, val.shape[-1])
        newval = np.full((len(val2D), length), neutral, dtype=val.dtype)
        newval[:, idx] = val2D
        n_per_pos[0, idx] += 1
        newvals.append(newval)
    res = func(n_per_pos, *newvals)
    res = np.where(n_per_pos == 0, np.nan, res)
    return res.reshape(vals[0].shape[:-1] + (-1,))


def mean_on_index(*args, length=None):
    """Computes the mean of arrays specified as indices and values.

    Parameters
    ----------
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, default=None
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    return apply_on_index(
        lambda n, *args: np.divide(sum(*args), n, where=(n != 0)), *args, length=length
    )


def safe_filename(file: Path) -> Path:
    """Make sure file can be build without overriding an other.

    If file already exists, returns a new filename with a number in between
    parenthesis. If the parent of the file doesn't exist, it is created.

    Raises
    ------
    FileExistsError
        If one of the parents of the file to create is an existing file
    """
    file = Path(file)
    # Build parent directories if needed
    if not file.parent.is_dir():
        print("Building parent directories")
        file.parent.mkdir(parents=True)
    # Change filename if it already exists
    if file.exists():
        original_file = file
        file_dups = 0
        while file.exists():
            file_dups += 1
            file = Path(
                file.parent, original_file.stem + f"({file_dups})" + file.suffix
            )
            # python3.9: file.with_stem(original_file.stem + f'({file_dups})')
        print(f"{original_file} exists, changing filename to {file}")
    return file


class PredSequenceDatasetRAM(Dataset):
    def __init__(
        self,
        seq,
        winsize,
        head_interval,
        head_crop=0,
        n_tracks=1,
        reverse=False,
        stride=1,
        offset=0,
        jump_stride=None,
        transform=train_pytorch.idx_to_onehot,
    ):
        # Check winsize, head_interval, stride, offset and jump_stride compatibilities
        if winsize <= 0:
            raise ValueError(f"winsize ({winsize}) must be non-negative")
        if (
            head_interval <= 0
            or head_interval % stride != 0
            or winsize % head_interval != 0
        ):
            raise ValueError(
                f"head_interval {head_interval} must be non-negative, "
                f"a multiple of stride ({stride}) and divide winsize ({winsize})"
            )
        if offset < 0 or offset >= stride:
            raise ValueError(
                f"offset ({offset}) must be positive and smaller than stride ({stride})"
            )
        self.winsize = winsize
        self.head_interval = head_interval
        self.head_crop = head_crop
        self.n_tracks = n_tracks
        self.reverse = reverse
        self.stride = stride
        self.offset = offset
        self.transform = transform
        self.slide_length = head_interval // stride
        if head_crop < 0 or head_crop >= (winsize // head_interval) / 2:
            raise ValueError(
                f"head_crop ({head_crop}) must be positive and smaller than half the number of heads ({winsize // head_interval})"
            )
        self.n_heads = winsize // head_interval - 2 * head_crop
        prediction_size = winsize - 2 * head_crop * head_interval
        if jump_stride is None:
            jump_stride = prediction_size
        elif (
            jump_stride <= 0
            or jump_stride > prediction_size
            or jump_stride % head_interval != 0
        ):
            raise ValueError(
                f"jump_stride {jump_stride} must be positive, "
                f"smaller than prediction size ({prediction_size}) "
                f"and a multiple of head_interval ({head_interval})"
            )
        self.jump_stride = jump_stride
        self.n_kept_heads = jump_stride // head_interval

        if reverse:
            # Reverse complement the sequences
            seq = train_pytorch.RC_idx(seq)
        # Last position where a window can be taken
        last_valid_pos = seq.shape[-1] - winsize
        # Last position where a full slide of window can be taken
        last_slide_start = last_valid_pos - head_interval + stride
        # Adjust last_slide_start to offset
        last_slide_start -= (last_slide_start - offset) % stride
        if last_valid_pos < 0:
            raise ValueError(
                f"Data length ({seq.shape[-1]}) must be longer than winsize ({winsize})"
            )
        if last_slide_start < offset:
            raise ValueError(
                f"No valid slide for length " f"{seq.shape[-1]} with offset {offset}"
            )
        slide = np.arange(0, head_interval, stride)
        slide_starts = np.arange(offset, last_slide_start + 1, jump_stride)
        # Add last possible slide_start (some values may be recomputed)
        if slide_starts[-1] < last_slide_start:
            slide_starts = np.append(slide_starts, last_slide_start)
            self.last_jump = (last_slide_start - offset) % jump_stride
        else:
            self.last_jump = jump_stride
        self.n_jumps = len(slide_starts)
        # Window positions for a single sequence
        indexes1d = (slide.reshape(1, -1) + slide_starts.reshape(-1, 1)).ravel()
        # If seq is multidimensionnal, consider all dimensions but the last
        # to be different sequences. Flatten the seq and propagate window
        # positions accross all sequences
        self.n_seqs = np.prod(np.array(seq.shape[:-1]), dtype=int)
        increments = np.arange(0, self.n_seqs * seq.shape[-1], seq.shape[-1])
        # Window positions for all sequences in the flattened seq
        self.positions = (indexes1d.reshape(1, -1) + increments.reshape(-1, 1)).ravel()
        self.original_shape = seq.shape
        self.seq = seq.ravel()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        pos = self.positions[idx]
        seq = self.seq[pos : pos + self.winsize]
        if self.transform:
            seq = self.transform(seq)
        return seq

    def reshaper(self, preds, kept_heads_start=None):
        # Shape (n_windows, n_heads, n_tracks)
        if kept_heads_start is not None:
            # Select only specific heads
            kept_heads_stop = kept_heads_start + self.n_kept_heads
            if kept_heads_start < 0 or kept_heads_stop > self.n_heads:
                raise ValueError(
                    f"kept_head_start ({kept_heads_start}) must be positive "
                    f"and smaller than n_heads - n_kept_heads ({self.n_heads - self.n_kept_heads})"
                )
            preds = preds[:, kept_heads_start:kept_heads_stop]
        elif self.n_kept_heads != self.n_heads:
            print(f"Keeping only first {self.n_kept_heads} out of {self.n_heads} heads")
            preds = preds[:, : self.n_kept_heads]
        # Reshape (n_windows, n_kept_heads, n_tracks)
        # => (n_seqs, n_jumps, slide_length, n_kept_heads, n_tracks)
        preds = preds.reshape(
            self.n_seqs,
            self.n_jumps,
            self.slide_length,
            self.n_kept_heads,
            self.n_tracks,
        )
        # Transpose slide_length and n_kept_heads before flattening them to
        # get proper sequence order, second to last dimension is pred_length_per_jump
        preds = np.transpose(preds, [0, 1, 3, 2, 4])
        preds = preds.reshape((self.n_seqs, self.n_jumps, -1, self.n_tracks))
        # Seperate last jump to truncate its beginning then put it back
        first_part = preds[:, :-1].reshape(self.n_seqs, -1, self.n_tracks)  # ndim=3
        last_part = preds[:, -1, -(self.last_jump // self.stride) :]
        preds = np.concatenate(
            [first_part, last_part], axis=1
        )  # shape (n_seqs, pred_length, n_tracks)
        # Put back in original shape, second to last dimension being pred_length
        preds = preds.reshape(self.original_shape[:-1] + (-1, self.n_tracks))
        if self.reverse:
            # Reverse predictions
            preds = np.flip(preds, axis=-2)
        return preds

    def get_indices(self, kept_heads_start=None):
        pred_start = self.head_crop * self.head_interval
        pred_stop = self.head_interval - 1 + self.head_crop * self.head_interval
        if kept_heads_start is not None:
            pred_start += kept_heads_start * self.head_interval
            pred_stop += (
                self.n_heads - self.n_kept_heads - kept_heads_start
            ) * self.head_interval
        elif self.n_kept_heads != self.n_heads:
            pred_stop += self.n_kept_heads * self.head_interval
        positions = np.arange(
            pred_start + self.offset, self.original_shape[-1] - pred_stop, self.stride
        )
        if self.reverse:
            positions = np.flip(self.original_shape[-1] - positions - 1)
        return positions


def predict(
    model,
    seq,
    winsize,
    head_interval,
    head_crop=0,
    n_tracks=1,
    reverse=False,
    stride=1,
    offset=0,
    jump_stride=None,
    kept_heads_start=None,
    batch_size=1024,
    num_workers=4,
    verbose=True,
):
    pred_dataset = PredSequenceDatasetRAM(
        seq,
        winsize,
        head_interval,
        head_crop=head_crop,
        n_tracks=n_tracks,
        reverse=reverse,
        stride=stride,
        offset=offset,
        jump_stride=jump_stride,
    )
    pred_dataloader = DataLoader(
        pred_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    res = []
    size = len(pred_dataset)
    with torch.no_grad():
        for batch, X in enumerate(pred_dataloader):
            X = X.to(next(model.parameters()).device)
            res.append(model(X).cpu().numpy())
            if verbose and batch % 100 == 0:
                print(f"{(batch+1)*len(X)}/{size}")
    res = np.concatenate(res, axis=0)
    return pred_dataset.reshaper(res, kept_heads_start), pred_dataset.get_indices(
        kept_heads_start
    )


if __name__ == "__main__":
    # Get arguments
    args = parsing()
    # Get cpu, gpu or mps device for training.
    args.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.verbose:
        print(f"Using {args.device} device")

    # Build output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    args.architecture = models.ARCHITECTURES[args.architecture]
    model = args.architecture(args.n_tracks).to(args.device)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    # Load fasta
    seq_dict = train_pytorch.ordinal_encoder(
        {
            res.id: res.seq
            for res in SeqIO.parse(args.fasta_file, "fasta")
            if args.chroms == "all" or res.id in args.chroms
        }
    )

    # Determine prediction options
    filename_suff = ""
    if args.middle:
        jump_stride = args.winsize // 2 - args.head_crop * args.head_interval
        kept_heads_start = args.winsize // (4 * args.head_interval) - args.head_crop
        filename_suff += "_mid"
    else:
        jump_stride = None
        kept_heads_start = None
    if args.strand in ["both", "merge"]:
        strands = ["for", "rev"]
        if args.strand == "merge":
            filename_suff += "_merge"
    else:
        strands = [args.strand]

    # Initialize output bigwigs
    bw_handles = []
    for track in range(args.n_tracks):
        filename = safe_filename(
            Path(
                args.output_dir,
                f"preds_{Path(args.model_file).stem}_on_{Path(args.fasta_file).stem}{filename_suff}_track{track}.bw",
            )
        )
        bw = pbw.open(str(filename), "w")
        if args.strand == "merge":
            bw.addHeader([(k, len(v)) for k, v in seq_dict.items()])
        else:
            bw.addHeader(
                [
                    (f"{k}_{strand}", len(v))
                    for k, v in seq_dict.items()
                    for strand in strands
                ]
            )
        bw_handles.append(bw)
    # Make predictions
    for chrom, seq in seq_dict.items():
        if args.verbose:
            print(f"Predicting on chromosome {chrom}")
        if args.strand == "merge":
            res = []
        for strand in strands:
            if args.verbose:
                print(f"Strand {strand}")
            pred, index = predict(
                model,
                seq,
                args.winsize,
                args.head_interval,
                head_crop=args.head_crop,
                n_tracks=args.n_tracks,
                jump_stride=jump_stride,
                kept_heads_start=kept_heads_start,
                reverse=(strand == "rev"),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            if args.strand == "merge":
                # Save for merge
                res.append((index, pred))
            else:
                # Write values immediately
                for track, bw in enumerate(bw_handles):
                    bw.addEntries(
                        f"{chrom}_{strand}", index, values=pred[..., track], span=1
                    )
        if args.strand == "merge":
            # Average forward and reverse predictions
            for track, bw in enumerate(bw_handles):
                pred = mean_on_index(
                    *[(index, pred[..., track]) for index, pred in res]
                )
                index = np.isfinite(pred).nonzero()[0]
                pred = pred[index]
                bw.addEntries(chrom, index, values=pred, span=1)
    bw.close()
