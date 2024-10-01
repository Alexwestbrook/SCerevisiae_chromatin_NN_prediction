#!/bin/env python

import argparse
from pathlib import Path
from typing import Tuple

import Bio
import models
import numpy as np
import shap
import torch
import utils
from Bio import SeqIO
from predict_pytorch import PredSequenceDatasetRAM
from torch.utils.data import DataLoader
from train_pytorch import SequenceDatasetRAM


def parsing() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Parameters
    ----------
    python command-line

    Returns
    -------
    argparse.Namespace
        Namespace with all arguments as attributes
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
        help="Fasta file containing genome to explain on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-l",
        "--label_files",
        help="Bigwig file containing labels per position in fasta",
        nargs="+",
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
        "-bc",
        "--background_chroms",
        help='Chromosomes to use as background. Specify "all" for all chromosomes (default: %(default)s)',
        nargs="+",
        type=str,
        default="all",
    )
    parser.add_argument(
        "-c",
        "--chroms",
        help='Chromosomes to explain. Specify "all" for all chromosomes (default: %(default)s)',
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
        help="Spacing between output head in case of multiple outputs "
        "(default: %(default)s)",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--stride",
        help="Stride to use for explanation (default: %(default)s)",
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


def get_shap_values(
    explainer: shap.explainers._deep.DeepExplainer,
    seq: np.ndarray,
    winsize: int,
    head_interval: int,
    head_crop: int = 0,
    n_tracks: int = 1,
    reverse: bool = False,
    stride: int = 1,
    offset: int = 0,
    jump_stride: int = None,
    kept_heads_start: int = None,
    batch_size: int = 1024,
    num_workers: int = 4,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
    for batch, X in enumerate(pred_dataloader):
        X = X.to(next(explainer.model.parameters()).device)
        res.append(explainer.shap_values(X, check_additivity=False))
        if verbose and batch % 100 == 0:
            print(f"{(batch+1)*len(X)}/{size}")
    res = np.concatenate(res, axis=0)
    return res, pred_dataset.positions
    # return pred_dataset.reshaper(res, kept_heads_start), pred_dataset.get_indices(
    #     kept_heads_start
    # )


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
    seq_dict = utils.ordinal_encoder(
        {
            res.id: res.seq
            for res in SeqIO.parse(args.fasta_file, "fasta")
            if args.chroms == "all" or res.id in args.chroms
        }
    )

    dataset = SequenceDatasetRAM(
        seq_file=args.fasta_file,
        label_files=args.label_files,
        chroms=args.background_chroms,
        winsize=2048,
        head_interval=16,
    )
    data = []
    for i in range(0, len(dataset), 2048):
        data.append(dataset[i][0])
    data = torch.tensor(np.stack(data, axis=0)).to(args.device)
    SHAP = shap.DeepExplainer(model, data)

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

    # # Initialize output bigwigs
    # bw_handles = []
    # for track in range(args.n_tracks):
    #     filename = utils.safe_filename(
    #         Path(
    #             args.output_dir,
    #             f"preds_{Path(args.model_file).parts[-2]}_on_{Path(args.fasta_file).stem}{filename_suff}_track{track}.bw",
    #         )
    #     )
    #     bw = pbw.open(str(filename), "w")
    #     if args.strand == "merge":
    #         bw.addHeader([(k, len(v)) for k, v in seq_dict.items()])
    #     else:
    #         bw.addHeader(
    #             [
    #                 (f"{k}_{strand}", len(v))
    #                 for k, v in seq_dict.items()
    #                 for strand in strands
    #             ]
    #         )
    #     bw_handles.append(bw)
    # Compute shap values
    for chrom, seq in seq_dict.items():
        if args.verbose:
            print(f"Explain chromosome {chrom}")
        if args.strand == "merge":
            res = []
        for strand in strands:
            if args.verbose:
                print(f"Strand {strand}")
            res, positions = get_shap_values(
                SHAP,
                seq,
                args.winsize,
                args.head_interval,
                stride=args.stride,
                head_crop=args.head_crop,
                n_tracks=args.n_tracks,
                jump_stride=jump_stride,
                kept_heads_start=kept_heads_start,
                reverse=(strand == "rev"),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            filename = utils.safe_filename(
                Path(
                    args.output_dir,
                    f"shap_{Path(args.model_file).parts[-2]}_on_{Path(args.fasta_file).stem}{filename_suff}_{chrom}_{strand}_{args.stride}.npz",
                )
            )
            np.savez_compressed(filename, res=res, positions=positions)
    #         if args.strand == "merge":
    #             # Save for merge
    #             res.append((index, pred))
    #         else:
    #             # Write values immediately
    #             for track, bw in enumerate(bw_handles):
    #                 bw.addEntries(
    #                     f"{chrom}_{strand}", index, values=pred[..., track], span=1
    #                 )
    #     if args.strand == "merge":
    #         # Average forward and reverse predictions
    #         for track, bw in enumerate(bw_handles):
    #             pred = utils.mean_on_index(
    #                 *[(index, pred[..., track]) for index, pred in res]
    #             )
    #             index = np.isfinite(pred).nonzero()[0]
    #             pred = pred[index]
    #             bw.addEntries(chrom, index, values=pred, span=1)
    # bw.close()
