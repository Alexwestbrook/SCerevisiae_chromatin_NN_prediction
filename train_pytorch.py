#!/bin/env python

import argparse
import datetime
import json
import socket
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import Bio
import numpy as np
import pyBigWig as pbw
import torch
from Bio import SeqIO
from sklearn.preprocessing import OrdinalEncoder
from torch import nn
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
        "-f",
        "--fasta_file",
        help="Fasta file containing genome to train on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-l",
        "--label_file",
        help="Bigwig file containing labels per position in fasta",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory name. Must not already exist",
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
        "-ct",
        "--chrom_train",
        help="Chromosomes to use for training",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-cv",
        "--chrom_valid",
        help="Chromosomes to use for validation",
        nargs="+",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--strand",
        help="Strand to perform training on, choose between 'for', 'rev' or "
        "'both' (default: %(default)s)",
        default="both",
        type=str,
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
        "-lr",
        "--learn_rate",
        help="Value for learning rate (default: %(default)s)",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of training loops over the dataset (default: %(default)s)",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of samples to use for training in parallel (default: %(default)s)",
        default=1024,
        type=int,
    )
    # parser.add_argument(
    #     "-ss", "--same_samples",
    #     help="Indicates to use the same sample at each epoch",
    #     action='store_true')
    parser.add_argument(
        "-mt",
        "--max_train",
        help="Maximum number of batches per epoch for training (default: %(default)s)",
        default=256,
        type=int,
    )
    parser.add_argument(
        "-mv",
        "--max_valid",
        help="Maximum number of batches per epoch for validation (default: %(default)s)",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        help="Number of parallel workers for preprocessing batches (default: %(default)s)",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-bal",
        "--balance",
        help="Indicates to balance weights inside batches",
        action="store_true",
    )
    parser.add_argument(
        "-nc",
        "--n_class",
        help="Number of classes to divide labels into for weighting (default: %(default)s)",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-r0",
        "--remove0s",
        help="Indicates to ignore 0-valued labels from training set",
        action="store_true",
    )
    parser.add_argument(
        "-rN",
        "--removeNs",
        help="Indicates to remove windows with Ns from training set",
        action="store_true",
    )
    parser.add_argument(
        "-loss",
        "--loss_fn",
        help="loss function for training (default: %(default)s)",
        default="mae_cor",
        type=str,
    )
    parser.add_argument(
        "-optim",
        "--optimizer_ctor",
        help="optimizer to use for training (default: %(default)s)",
        default="adam",
        type=str,
    )
    # parser.add_argument(
    #     "-r", "--remove_indices",
    #     help="Npz archive containing indices of labels to remove from "
    #          "training set, with one array per chromosome",
    #     default=None,
    #     type=str)
    # parser.add_argument(
    #     "--seed",
    #     help="Seed to use for random shuffling of training samples",
    #     default=None,
    #     type=int)
    # parser.add_argument(
    #     "-da", "--disable_autotune",
    #     action='store_true',
    #     help="Indicates not to use earlystopping.")
    # parser.add_argument(
    #     "-p", "--patience",
    #     help="Number of epochs without improvement to wait before stopping "
    #          "training (default: %(default)s)",
    #     default=6,
    #     type=int)
    # parser.add_argument(
    #     "-dist", "--distribute",
    #     action='store_true',
    #     help="Indicates to use both GPUs with MirrorStrategy.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="0 for silent, 1 for progress bar and 2 for single line",
        default=2,
        type=int,
    )
    args = parser.parse_args()
    # Check that files exist
    for argname in ["fasta_file", "model_file"]:
        filename = vars(args)[argname]
        if not Path(filename).is_file():
            raise ValueError(
                f"File {filename} does not exist. Please enter a valid {argname}"
            )
    # Format chromosome names
    genome_name = Path(args.fasta_file).stem
    if genome_name == "W303":
        args.chrom_train = ["chr" + format(int(c), "02d") for c in args.chrom_train]
        args.chrom_valid = ["chr" + format(int(c), "02d") for c in args.chrom_valid]
    elif genome_name == "mm10":
        args.chrom_train = [f"chr{c}" for c in args.chrom_train]
        args.chrom_valid = [f"chr{c}" for c in args.chrom_valid]
    elif genome_name == "W303_Mmmyco":
        args.chrom_train = [f"chr{c}" if c != "Mmmyco" else c for c in args.chrom_train]
        args.chrom_valid = [f"chr{c}" if c != "Mmmyco" else c for c in args.chrom_valid]
    return args


def ordinal_encoder(
    inp: Union[Dict[str, str], List[str], str, Bio.Seq.Seq],
) -> Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]:
    # Format input
    if isinstance(inp, list):
        seqs = inp
    elif isinstance(inp, dict):
        seqs = list(inp.values())
    else:
        seqs = inp
    # Encode sequences
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=4, dtype=np.int8
    )
    encoder.fit(np.array(list("ACGT")).reshape((-1, 1)))
    seqs = [
        encoder.transform(np.array(list(seq)).reshape((-1, 1))).ravel() for seq in seqs
    ]
    # Format output
    if isinstance(inp, dict):
        seqs = dict(zip(inp.keys(), seqs))
    elif not isinstance(inp, list):
        seqs = seqs[0]
    return seqs


def idx_to_onehot(idx: np.ndarray[int], dtype: type = np.float32) -> np.ndarray:
    return np.eye(5, dtype=dtype)[:, :-1][idx]


def RC_idx(idx: np.ndarray[int]) -> np.ndarray[int]:
    # Copy data array before modification
    idx = idx.copy()
    idx[idx == 0] = -1
    idx[idx == 3] = 0
    idx[idx == -1] = 3
    idx[idx == 1] = -1
    idx[idx == 2] = 1
    idx[idx == -1] = 2
    return np.flip(idx, axis=-1)


def slicer_on_axis(
    arr: np.ndarray,
    slc: Union[slice, Iterable[slice]],
    axis: Union[None, int, Iterable[int]] = None,
) -> Tuple[slice]:
    """Build slice of array along specified axis.

    This function can be used to build slices from unknown axis parameter.

    Parameters
    ----------
    arr: ndarray
        Input array
    slc: slice or iterable of slices
        Slices of the array to take.
    axis: None or int or iterable of ints
        Axis along which to perform slicing. The default (None) is to take slices along first len(`slc`) dimensions.

    Returns
    -------
    tuple of slices
        Full tuple of slices to use to slice array

    Examples
    --------
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> arr
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> arr[slicer_on_axis(arr, slice(1, 3), axis=-1)]  # simple slice on last axis
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]],

           [[13, 14],
            [17, 18],
            [21, 22]]])
    >>> for axis, res in enumerate([arr[1:], arr[:, 1:], arr[:, :, 1:]]):  # unknown axis parameter
    >>>     print(np.all(arr[slicer_on_axis(arr, slice(1, None), axis=axis)] == res))
    True
    True
    True
    >>> arr[slicer_on_axis(arr, slice(None, -1))]  # no axis parameter
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)], axis=[0, 2])]  # multiple slices and axis
    array([[[ 1,  2],
            [ 5,  6],
            [ 9, 10]]])
    >>> arr[slicer_on_axis(arr, [slice(None, -1), slice(1, 3)])]  # multiple slices without axis parameter
    array([[[ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])
    >>> arr[slicer_on_axis(arr, slice(1, None), axis=[1, 2])]  # single slice on multiple axis
    array([[[ 5,  6,  7],
            [ 9, 10, 11]],

           [[17, 18, 19],
            [21, 22, 23]]])
    """
    full_slice = [slice(None)] * arr.ndim
    if isinstance(slc, slice):
        if axis is None:
            axis = 0
        if isinstance(axis, int):
            full_slice[axis] = slc
        else:
            for ax in axis:
                full_slice[ax] = slc
    else:
        if axis is None:
            axis = list(range(len(slc)))
        elif not isinstance(axis, Iterable):
            raise ValueError("if slc is an iterable, axis must be an iterable or None")
        elif len(axis) != len(slc):
            raise ValueError("axis and slc must have same length")
        for s, ax in zip(slc, axis):
            if full_slice[ax] != slice(None):
                raise ValueError("Can't set slice on same axis twice")
            full_slice[ax] = s
    return tuple(full_slice)


def moving_sum(arr: np.ndarray, n: int, axis: Union[None, int] = None) -> np.ndarray:
    """Compute moving sum of array

    Parameters
    ----------
    arr: ndarray
        Input array
    n: int
        Length of window to compute sum on, must be greater than 0
    axis: None or int, optional
        Axis along which the moving sum is computed, the default (None) is to compute the moving sum over the flattened array.

    Returns
    -------
    ndarray
        Array of moving sum, with size along `axis` reduced by `n`-1.

    Examples
    --------
    >>> moving_sum(np.arange(10), n=2)
    array([ 1,  3,  5,  7,  9, 11, 13, 15, 17])
    >>> arr = np.arange(24).reshape(2, 3, 4)
    >>> moving_sum(arr, n=2, axis=-1)
    array([[[ 1,  3,  5],
            [ 9, 11, 13],
            [17, 19, 21]],

           [[25, 27, 29],
            [33, 35, 37],
            [41, 43, 45]]])
    """
    if n <= 0:
        raise ValueError(f"n must be greater than 0, but is equal to {n}")
    elif axis is None and n > arr.size:
        raise ValueError(
            f"Can't compute moving_sum of {n} values on flattened array with length {arr.size}"
        )
    elif axis is not None and n > arr.shape[axis]:
        raise ValueError(
            f"Can't compute moving_sum of {n} values on axis {axis} with length {arr.shape[axis]}"
        )
    res = np.cumsum(arr, axis=axis)
    res[slicer_on_axis(res, slice(n, None), axis=axis)] = (
        res[slicer_on_axis(res, slice(n, None), axis=axis)]
        - res[slicer_on_axis(res, slice(None, -n), axis=axis)]
    )
    return res[slicer_on_axis(res, slice(n - 1, None), axis=axis)]


class SequenceDatasetRAM(Dataset):
    def __init__(
        self,
        seq_file,
        label_file,
        chroms,
        winsize,
        head_interval,
        strand="both",
        removeNs=True,
        remove0s=True,
        transform=idx_to_onehot,
        target_transform=None,
        rng=np.random.default_rng(),
    ):
        # Remove duplicate chromosomes
        if len(set(chroms)) != len(chroms):
            chroms = list(set(chroms))
        self.chroms = chroms
        self.winsize = winsize
        self.head_interval = head_interval
        self.train = train
        self.strand = strand
        self.transform = transform
        self.target_transform = target_transform
        self.rng = rng
        # Load and check chromosomes
        self.seq_dict = {
            res.id: res.seq
            for res in SeqIO.parse(seq_file, "fasta")
            if res.id in chroms
        }
        no_seq_found = set(chroms) - self.seq_dict.keys()
        if no_seq_found:
            raise ValueError(f"Couldn't find sequence for {no_seq_found}")
        # Load and check labels
        self.labels_dict = {}
        label_range = None
        with pbw.open(label_file) as bw:
            for i, chrom in enumerate(chroms):
                length = bw.chroms(chrom)
                if length is None:
                    raise ValueError(f"{chrom} not in {label_file}")
                self.labels_dict[i] = bw.values(chrom, 0, -1, numpy=True)
                # Determine minimum and maximum values for labels
                if label_range is None:
                    label_range = (
                        np.min(self.labels_dict[i]),
                        np.max(self.labels_dict[i]),
                    )
                else:
                    label_range = (
                        min(label_range[0], np.min(self.labels_dict[i])),
                        max(label_range[1], np.max(self.labels_dict[i])),
                    )
        self.label_range = label_range
        # Encode chromosome ids and nucleotides as integers
        self.seq_dict = ordinal_encoder(
            {i: self.seq_dict[chrom] for i, chrom in enumerate(chroms)}
        )
        # Extract valid positions
        self.positions = []
        for i, seq in enumerate(self.seq_dict.values()):
            indexes = np.ma.arange(len(seq) - winsize + 1)
            indexes[-winsize + 1 :] = np.ma.masked
            if removeNs:
                # Remove windows containing an N
                N_in_window = moving_sum(seq == 4, winsize).astype(bool)
                indexes[N_in_window] = np.ma.masked
            if remove0s:
                full_0_window = moving_sum(self.labels_dict[i] == 0, winsize) == winsize
                indexes[full_0_window] = np.ma.masked
            indexes = indexes.compressed()
            self.positions.extend(list(zip([i] * len(indexes), indexes)))
        # Initialize weights
        self.weights_dict = {}
        for i, labels in enumerate(self.labels_dict.values()):
            weights = np.ones(len(labels), dtype=np.int8)
            if remove0s:
                # Weight labels to ignore to 0
                weights[labels == 0] = 0
            self.weights_dict[i] = weights

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        chrom, pos = self.positions[idx]
        seq = self.seq_dict[chrom][pos : pos + self.winsize]
        if (self.strand == "both" and self.rng.random(1) > 0.5) or self.strand == "rev":
            # Take reverse sequence
            seq = RC_idx(seq)
            label = self.labels_dict[chrom][
                pos + self.winsize - 1 : pos - 1
                if pos > 0
                else None : -self.head_interval
            ].copy()
        else:
            label = self.labels_dict[chrom][
                pos : pos + self.winsize : self.head_interval
            ]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)
        weight = self.weights_dict[chrom][pos : pos + self.winsize : self.head_interval]
        return seq, label, weight


def collate_samplebalance(batch, n_class=100, range=(0, 1)):
    seq, label, weight = tuple(np.array(x) for x in zip(*batch))
    flat_label = label.flatten()
    flat_weight = weight.flatten()
    label_eff = flat_label[flat_weight != 0]  # number nonzero weight labels
    # Divide [0, 1] into bins and determine bin_idx for each label
    bin_values, bin_edges = np.histogram(label_eff, bins=n_class, range=range)
    n_class_eff = np.sum(bin_values != 0)
    bin_idx = np.digitize(flat_label, bin_edges)
    bin_idx[bin_idx == n_class + 1] = n_class
    bin_idx -= 1
    # Compute balanced sample weights from bin occupancy
    new_weight = np.zeros_like(flat_weight, dtype=np.float64)
    new_weight = np.divide(
        len(label_eff) / n_class_eff,
        bin_values[bin_idx],
        out=new_weight,
        where=flat_weight != 0,
    )
    # Reshape as original
    new_weight = new_weight.reshape(weight.shape)
    return tuple(torch.Tensor(x) for x in (seq, label, new_weight))


class MnaseEtienneNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv1d(4, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Conv1d(64, 16, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Conv1d(16, 80, 8, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(80 * 250, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class ResidualConcatLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class BassenjiEtienneNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            self.pooled_conv_wrapper(4, 32, 12, pool_size=8),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=4),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=4),
            self.dilated_conv_wrapper(32, 16, 5, dilation=2),
            ResidualConcatLayer(self.dilated_conv_wrapper(16, 16, 5, dilation=4)),
            ResidualConcatLayer(self.dilated_conv_wrapper(32, 16, 5, dilation=8)),
            ResidualConcatLayer(self.dilated_conv_wrapper(48, 16, 5, dilation=16)),
            nn.Conv1d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def pooled_conv_wrapper(self, in_channels, out_channels, kernel_size, pool_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def dilated_conv_wrapper(self, in_channels, out_channels, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


class BassenjiMnaseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            self.pooled_conv_wrapper(4, 32, 12, pool_size=4),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=2),
            self.pooled_conv_wrapper(32, 32, 5, pool_size=2),
            self.dilated_conv_wrapper(32, 16, 5, dilation=2),
            ResidualConcatLayer(self.dilated_conv_wrapper(16, 16, 5, dilation=4)),
            ResidualConcatLayer(self.dilated_conv_wrapper(32, 16, 5, dilation=8)),
            ResidualConcatLayer(self.dilated_conv_wrapper(48, 16, 5, dilation=16)),
            nn.Conv1d(64, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

    def pooled_conv_wrapper(self, in_channels, out_channels, kernel_size, pool_size):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def dilated_conv_wrapper(self, in_channels, out_channels, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return self.conv_stack(x)


def torch_mae_cor(
    output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None
) -> torch.Tensor:
    """Compute loss with MAE and correlation."""
    if weight is not None:
        weight = weight.flatten()
        target = target.flatten()[weight != 0]
        output = output.flatten()[weight != 0]
        weight = weight[weight != 0]
    else:
        weight = 1

    X = target - torch.mean(weight * target)
    Y = output - torch.mean(weight * output)
    sigma_XY = torch.sum(weight * X * Y)
    sigma_X = torch.sum(weight * X * X)
    sigma_Y = torch.sum(weight * Y * Y)

    cor = sigma_XY / (torch.sqrt(sigma_X * sigma_Y) + 1e-45)
    mae = torch.mean(weight * torch.abs(target - output))
    return (1 - cor) + mae


def train(dataloader, model, loss_fn, optimizer, device, max_batch_per_epoch):
    size = len(dataloader.dataset)
    if max_batch_per_epoch:
        size = min(size, len(next(iter(dataloader))[0]) * max_batch_per_epoch)
    model.train()
    for batch, (X, y, w) in enumerate(dataloader):
        if max_batch_per_epoch and batch >= max_batch_per_epoch:
            break
        X, y, w = X.to(device), y.to(device), w.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y, w)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device, max_batch_per_epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (X, y, w) in enumerate(dataloader):
            if max_batch_per_epoch and batch >= max_batch_per_epoch:
                break
            X, y, w = X.to(device), y.to(device), w.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y, w).item()
    test_loss /= batch
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def main(args):
    training_dataRAM = SequenceDatasetRAM(
        args.fasta_file,
        args.label_file,
        args.chrom_train,
        winsize=args.winsize,
        head_interval=args.head_interval,
        strand=args.strand,
        remove0s=args.remove0s,
        removeNs=args.removeNs,
        transform=idx_to_onehot,
    )
    valid_dataRAM = SequenceDatasetRAM(
        args.fasta_file,
        args.label_file,
        args.chrom_valid,
        winsize=args.winsize,
        head_interval=args.head_interval,
        strand=args.strand,
        remove0s=args.remove0s,
        removeNs=args.removeNs,
        transform=idx_to_onehot,
    )
    if args.balance:
        print("label range: ", training_dataRAM.label_range)

        def collate_fn(batch):
            return collate_samplebalance(
                batch, args.n_class, training_dataRAM.label_range
            )
    else:
        collate_fn = None
    train_dataloaderRAM = DataLoader(
        training_dataRAM,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_dataloaderRAM = DataLoader(
        valid_dataRAM,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Define model and optimizer
    model = args.architecture().to(args.device)
    optimizer = args.optimizer_ctor(model.parameters(), args.learn_rate)

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(
            train_dataloaderRAM,
            model,
            args.loss_fn,
            optimizer,
            device=args.device,
            max_batch_per_epoch=args.max_train,
        )
        test(valid_dataloaderRAM, model, args.loss_fn, args.device, args.max_valid)
    print("Done!")

    model_file = Path(args.output_dir, "model_state.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Saved PyTorch Model State to {model_file}")


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
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
    print(f"Using {args.device} device")

    # Build output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    # Store arguments in config file
    with open(Path(args.output_dir, "config.json"), "w") as f:
        json.dump(
            {
                **vars(args),
                **{
                    "timestamp": str(tmstmp),
                    "machine": socket.gethostname(),
                },
            },
            f,
            indent=4,
        )
        f.write("\n")
    # Convert to non json serializable objects
    architectures = {
        "BassenjiMnaseNetwork": BassenjiMnaseNetwork,
        "BassenjiEtienneNetwork": BassenjiEtienneNetwork,
    }
    args.architecture = architectures[args.architecture]
    losses = {"mae_cor": torch_mae_cor}
    args.loss_fn = losses[args.loss_fn]
    optimizer_ctors = {"adam": torch.optim.Adam}
    args.optimizer_ctor = optimizer_ctors[args.optimizer_ctor]

    # Start computations, save total time even if there was a failure
    try:
        main(args)
    except:
        with open(Path(args.output_dir, "train_time.txt"), "w") as f:
            f.write("Aborted\n")
            f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")
        raise
    with open(Path(args.output_dir, "train_time.txt"), "w") as f:
        f.write(f"total time: {datetime.datetime.now() - tmstmp}\n")