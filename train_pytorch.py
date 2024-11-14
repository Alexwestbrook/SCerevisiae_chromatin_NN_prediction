#!/bin/env python

import argparse
import datetime
import json
import socket
from pathlib import Path
from typing import Callable, List, Tuple

import Bio
import models
import numpy as np
import pyBigWig as pbw
import torch
import utils
from Bio import SeqIO
from numpy.typing import ArrayLike
from scipy.signal import convolve
from torch import nn
from torch.utils.data import DataLoader, Dataset

# torch.autograd.set_detect_anomaly(True)


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
        "-f",
        "--fasta_file",
        help="Fasta file containing genome to train on",
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
        "-crop",
        "--head_crop",
        help="Number of heads to crop on left and right side of window (default: %(default)s)",
        type=int,
        default=0,
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
    parser.add_argument(
        "-p",
        "--patience",
        help="Number of epochs without improvement to wait before stopping "
        "training (default: %(default)s)",
        default=6,
        type=int,
    )
    # parser.add_argument(
    #     "-dist", "--distribute",
    #     action='store_true',
    #     help="Indicates to use both GPUs with MirrorStrategy.")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Indicates to show information messages",
        action="store_true",
    )
    args = parser.parse_args()
    # Check that files exist
    for filename in [args.fasta_file] + args.label_files:
        if not Path(filename).is_file():
            raise ValueError(f"File {filename} does not exist.")
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


class SequenceDatasetRAM(Dataset):
    def __init__(
        self,
        seq_file: str,
        label_files: List[str],
        chroms: List[str],
        winsize: int,
        head_interval: int,
        head_crop: int = 0,
        strand: str = "both",
        removeNs: bool = True,
        remove0s: bool = True,
        transform: Callable = utils.idx_to_onehot,
        target_transform: Callable = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        # Remove duplicate chromosomes
        if len(set(chroms)) != len(chroms):
            chroms = list(set(chroms))
        self.chroms = chroms
        self.winsize = winsize
        self.head_interval = head_interval
        head_indices = np.arange(0, winsize, head_interval)
        if head_crop > 0:
            head_indices = head_indices[head_crop:-head_crop]
        self.head_mask = np.zeros(winsize, dtype=bool)
        self.head_mask[head_indices] = True
        self.n_heads = np.sum(self.head_mask)
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
        if isinstance(label_files, str):
            label_files = [label_files]
        self.labels_dict = {i: [] for i in range(len(chroms))}
        self.label_ranges = []
        for label_file in label_files:
            label_range = None
            with pbw.open(label_file) as bw:
                for i, chrom in enumerate(chroms):
                    length = bw.chroms(chrom)
                    if length is None:
                        raise ValueError(f"{chrom} not in {label_file}")
                    self.labels_dict[i].append(bw.values(chrom, 0, -1, numpy=True))
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
            self.label_ranges.append(label_range)
        for i, chrom in enumerate(chroms):
            self.labels_dict[i] = np.stack(self.labels_dict[i], axis=1)
        # Encode chromosome ids and nucleotides as integers
        self.seq_dict = utils.ordinal_encoder(
            {i: self.seq_dict[chrom] for i, chrom in enumerate(chroms)}
        )
        # Extract valid positions
        self.positions = []
        for i, seq in enumerate(self.seq_dict.values()):
            indexes = np.ma.arange(len(seq) - winsize + 1)
            indexes[-winsize + 1 :] = np.ma.masked
            if removeNs:
                # Remove windows containing an N
                N_in_window = utils.moving_sum(seq == 4, winsize).astype(bool)
                indexes[N_in_window] = np.ma.masked
            if remove0s:
                full_0_position = np.all(self.labels_dict[i] == 0, axis=-1)
                # convolve will flip the second array, convert to int for integer output
                full_0_window = (
                    convolve(
                        full_0_position,
                        self.head_mask[::-1].astype(int),
                        mode="valid",
                    )
                    == self.n_heads
                )
                indexes[full_0_window] = np.ma.masked
            indexes = indexes.compressed()
            self.positions.extend(list(zip([i] * len(indexes), indexes)))
        # Initialize weights
        self.weights_dict = {}
        for i, labels in enumerate(self.labels_dict.values()):
            weights = np.ones_like(labels, dtype=bool)
            if remove0s:
                # Weight labels to ignore to 0
                weights[labels == 0] = 0
            self.weights_dict[i] = weights

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chrom, pos = self.positions[idx]
        seq = self.seq_dict[chrom][pos : pos + self.winsize]
        label = self.labels_dict[chrom][pos : pos + self.winsize]
        weight = self.weights_dict[chrom][pos : pos + self.winsize]
        if (self.strand == "both" and self.rng.random(1) > 0.5) or self.strand == "rev":
            # Take reverse sequence
            seq = utils.RC_idx(seq)
            label = label[self.head_mask[::-1]][::-1].copy()
            weight = weight[self.head_mask[::-1]][::-1].copy()
        else:
            label = label[self.head_mask]
            weight = weight[self.head_mask]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)
        return seq, label, weight


def balance_label_weights(
    label: np.ndarray,
    has_weight: np.ndarray,
    n_class: int = 100,
    label_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Balance weights by dividing labels into bins.

    Balancing is done on the flattened array.

    Parameters
    ----------
    label : np.ndarray
        Array of labels to weight
    has_weight : np.ndarray
        Boolean mask of same shape as label, indicating where to compute weights.
        Other locations will have weights of 0.
    n_class : int, optional
        Number of classes to divide the range into
    label_range : tuple, optional
        Range of the bins for binning labels

    Returns
    -------
    np.ndarray
        Array of balanced weights of same shape as label.
    """
    # Flatten input arrays
    original_shape = label.shape
    label = label.flatten()
    has_weight = has_weight.flatten()
    label_eff = label[has_weight]  # number nonzero weight labels
    # Divide label range into bins and determine bin_idx for each label
    bin_values, bin_edges = np.histogram(label_eff, bins=n_class, range=label_range)
    n_class_eff = np.sum(bin_values != 0)
    bin_idx = np.digitize(label, bin_edges)
    bin_idx[bin_idx == n_class + 1] = n_class
    bin_idx -= 1
    # Compute balanced sample weights from bin occupancy
    weight = np.zeros_like(has_weight, dtype=np.float64)
    return np.divide(
        len(label_eff) / n_class_eff,
        bin_values[bin_idx],
        out=weight,
        where=has_weight,
    ).reshape(original_shape)


def collate_samplebalance(
    batch: Tuple[ArrayLike, ArrayLike, ArrayLike],
    n_class: int = 100,
    label_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq, label, weight = tuple(np.array(x) for x in zip(*batch))
    if label.ndim == 3:
        # shape (batch_size, outputs, tracks)
        new_weight = np.empty_like(weight)
        n_classes = utils.make_iterator(n_class, label.shape[-1])
        label_ranges = utils.make_iterator(label_range, label.shape[-1])
        for j in range(label.shape[-1]):
            n_class = next(n_classes)
            label_range = next(label_ranges)
            # balance accross batch and outputs
            new_weight[..., j] = balance_label_weights(
                label[..., j],
                weight[..., j] != 0,
                n_class=n_class,
                label_range=label_range,
            ).reshape(weight[..., j].shape)
    else:
        # shape (batch_size, outputs)
        new_weight = balance_label_weights(
            label,
            weight != 0,
            n_class=n_class,
            label_range=label_range,
        ).reshape(weight.shape)
    return tuple(torch.Tensor(x) for x in (seq, label, new_weight))


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


def torch_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    X = x - torch.mean(x)
    Y = y - torch.mean(y)
    sigma_XY = torch.sum(X * Y)
    sigma_X = torch.sum(X * X)
    sigma_Y = torch.sum(Y * Y)

    return sigma_XY / (torch.sqrt(sigma_X * sigma_Y) + 1e-45)


def weighted_mse_loss(
    output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.mean(weight * (output - target) ** 2)


def torch_loss_bytrack(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor = None,
    loss_fn: Callable = torch_mae_cor,
) -> torch.Tensor:
    """Compute loss on multiple tracks."""

    track_losses = []
    for track in range(output.shape[-1]):
        if torch.any(weight[..., track] != 0):
            track_losses.append(
                loss_fn(output[..., track], target[..., track], weight[..., track])
            )
    return torch.stack(track_losses)


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_batch_per_epoch: int,
) -> float:
    size = len(dataloader.dataset)
    if max_batch_per_epoch:
        size = min(size, len(next(iter(dataloader))[0]) * max_batch_per_epoch)
    model.train()
    avg_loss = 0
    for batch, (X, y, w) in enumerate(dataloader):
        if max_batch_per_epoch and batch >= max_batch_per_epoch:
            break
        X, y, w = X.to(device), y.to(device), w.to(device)

        # Compute prediction error
        pred = model(X)
        loss = torch_loss_bytrack(pred, y, w, loss_fn).mean()

        # Backpropagation
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()
        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss /= batch
    print(f"Avg train loss: {avg_loss:>8f} \n")
    return avg_loss


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    device: str,
    max_batch_per_epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    n_tracks = next(iter(dataloader))[1].shape[-1]
    test_loss_bytrack = torch.zeros(n_tracks).to(device)
    test_cor_bytrack = torch.zeros(n_tracks).to(device)
    with torch.no_grad():
        for batch, (X, y, w) in enumerate(dataloader):
            if max_batch_per_epoch and batch >= max_batch_per_epoch:
                break
            X, y, w = X.to(device), y.to(device), w.to(device)
            pred = model(X)
            test_loss_bytrack += torch_loss_bytrack(pred, y, w, loss_fn)
            for track in range(n_tracks):
                test_cor_bytrack[track] += torch_correlation(
                    pred[..., track], y[..., track]
                )
    test_loss_bytrack /= batch
    test_cor_bytrack /= batch
    print(
        f"Avg test loss: {test_loss_bytrack.mean().item():>8f}\tAvg test correlation: {test_cor_bytrack.mean().item():>8f}\n"
    )
    for track in range(n_tracks):
        print(
            f"Track {track}\tAvg test loss: {test_loss_bytrack[track]:>8f}\tAvg test correlation: {test_cor_bytrack[track]:>8f}"
        )
    print()
    return test_loss_bytrack, test_cor_bytrack


def main(args: argparse.Namespace) -> None:
    training_data = SequenceDatasetRAM(
        args.fasta_file,
        args.label_files,
        args.chrom_train,
        winsize=args.winsize,
        head_interval=args.head_interval,
        head_crop=args.head_crop,
        strand=args.strand,
        remove0s=args.remove0s,
        removeNs=args.removeNs,
        transform=utils.idx_to_onehot,
    )
    valid_data = SequenceDatasetRAM(
        args.fasta_file,
        args.label_files,
        args.chrom_valid,
        winsize=args.winsize,
        head_interval=args.head_interval,
        head_crop=args.head_crop,
        strand=args.strand,
        remove0s=args.remove0s,
        removeNs=args.removeNs,
        transform=utils.idx_to_onehot,
    )
    if args.balance:
        print("label range: ", training_data.label_ranges)

        def collate_fn(batch):
            return collate_samplebalance(
                batch, args.n_class, training_data.label_ranges
            )
    else:
        collate_fn = None
    train_dataloader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Define model and optimizer
    model = args.architecture(n_tracks=len(args.label_files)).to(args.device)
    optimizer = args.optimizer_ctor(model.parameters(), args.learn_rate)

    model_file = Path(args.output_dir, "model_state.pt")
    best_valid_loss = np.inf
    wait = 0
    log_file = Path(args.output_dir, "epoch_logs.csv")
    with open(log_file, "w") as f:
        f.write(
            "epoch\ttrain_loss\tvalid_loss\tvalid_loss_bytrack\tvalid_cor_bytrack\n"
        )
    size = len(train_dataloader.dataset)
    if args.max_train:
        size = min(size, len(next(iter(train_dataloader))[0]) * args.max_train)
    step = 0
    last_val_step = 0
    train_loss = 0
    model.train()
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        for batch, (X, y, w) in enumerate(train_dataloader):
            X, y, w = X.to(args.device), y.to(args.device), w.to(args.device)

            # Compute prediction error
            pred = model(X)
            loss = torch_loss_bytrack(pred, y, w, args.loss_fn).mean()

            # Backpropagation
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            if batch % 50 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            step += 1

            # Evaluation
            if args.max_train and step % args.max_train:
                train_loss /= step - last_val_step
                print(f"Avg train loss: {train_loss:>8f} \n")
                valid_loss_bytrack, valid_cor_bytrack = test(
                    valid_dataloader,
                    model,
                    args.loss_fn,
                    args.device,
                    args.max_valid,
                )
                valid_loss = valid_loss_bytrack.mean().item()
                with open(log_file, "a") as f:
                    f.write(
                        f"{t}\t{train_loss:>8f}\t{valid_loss:>8f}\t{valid_loss_bytrack.cpu().numpy()}\t{valid_cor_bytrack.cpu().numpy()}\n"
                    )
                # Earlystopping
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), model_file)
                    wait = 0
                else:
                    wait += 1
                    if wait >= args.patience:
                        break
                # Reinitialize for training steps
                model.train()
                train_loss = 0
                last_val_step = step
    print("Done!")
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
    args.architecture = models.ARCHITECTURES[args.architecture]
    losses = {
        "mae_cor": torch_mae_cor,
        "mse": weighted_mse_loss,
        "PoissonNLL": nn.PoissonNLLLoss(),
    }
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
