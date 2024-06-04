# import argparse

from typing import Dict, List, Union

import Bio
import numpy as np
import pyBigWig as pbw
import torch
from Bio import SeqIO
from sklearn.preprocessing import OrdinalEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


class SequenceDatasetRAM(Dataset):
    def __init__(
        self,
        seq_file,
        label_file,
        chroms,
        winsize,
        head_interval,
        train=True,
        transform=idx_to_onehot,
        target_transform=None,
    ):
        # Remove duplicate chromosomes
        if len(set(chroms)) != len(chroms):
            chroms = list(set(chroms))
        self.chroms = chroms
        self.train = train
        self.head_interval = head_interval
        self.transform = transform
        self.target_transform = target_transform
        self.winsize = winsize
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
        with pbw.open(label_file) as bw:
            for i, chrom in enumerate(chroms):
                length = bw.chroms(chrom)
                if length is None:
                    raise ValueError(f"{chrom} not in {label_file}")
                self.labels_dict[i] = bw.values(chrom, 0, -1, numpy=True)
        # Encode chromosome ids and nucleotides as integers
        self.seq_dict = ordinal_encoder(
            {i: self.seq_dict[chrom] for i, chrom in enumerate(chroms)}
        )
        # Extract valid positions
        self.positions = []
        for i, seq in enumerate(self.seq_dict.values()):
            indexes = np.ma.arange(len(seq) - winsize + 1)
            indexes[-winsize + 1 :] = np.ma.masked
            N_in_window = np.convolve(
                seq == 4, np.ones(self.winsize, dtype=bool), mode="valid"
            )
            indexes[N_in_window] = np.ma.masked
            indexes = indexes.compressed()
            self.positions.extend(list(zip([i] * len(indexes), indexes)))
        if self.train:
            # Weight labels to ignore to 0
            self.weights_dict = {}
            for i, labels in enumerate(self.labels_dict.values()):
                weights = np.ones(len(labels), dtype=np.int8)
                weights[labels == 0] = 0
                self.weights_dict[i] = weights

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        chrom, pos = self.positions[idx]
        seq = self.seq_dict[chrom][pos : pos + self.winsize]
        label = self.labels_dict[chrom][pos : pos + self.winsize : self.head_interval]
        if self.transform:
            seq = self.transform(seq)
        if self.target_transform:
            label = self.target_transform(label)
        if self.train:
            weight = self.weights_dict[chrom][
                pos : pos + self.winsize : self.head_interval
            ]
            return seq, label, weight
        else:
            return seq, label


def collate_samplebalance(batch, n_class=500):
    seq, label, weight = tuple(np.array(x) for x in zip(*batch))
    flat_label = label.flatten()
    flat_weight = weight.flatten()
    label_eff = flat_label[flat_weight != 0]  # number nonzero weight labels
    # Divide [0, 1] into bins and determine bin_idx for each label
    bin_values, bin_edges = np.histogram(label_eff, bins=n_class, range=(0, 1))
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


def train(dataloader, model, loss_fn, optimizer, max_batch_per_epoch):
    size = len(dataloader.dataset)
    if max_batch_per_epoch:
        size = min(size, len(next(iter(dataloader))[0]) * max_batch_per_epoch)
    model.train()
    for batch, (X, y, w) in enumerate(dataloader):
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
        if max_batch_per_epoch and batch > max_batch_per_epoch:
            break


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    fasta_file = "/home/alex/shared_folder/SCerevisiae/genome/W303_Mmmyco.fa"
    nuc_file = "/home/alex/shared_folder/SCerevisiae/data/labels_myco_nuc.bw"
    pol_file = (
        "/home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_pol_ratio.bw"
    )
    coh_file = (
        "/home/alex/shared_folder/SCerevisiae/data/GSE217022/labels_myco_coh_ratio.bw"
    )

    training_dataRAM = SequenceDatasetRAM(
        fasta_file,
        nuc_file,
        [
            f"chr{num}"
            for num in [
                "I",
                "II",
                "III",
                "IV",
                "V",
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
                "XI",
                "XII",
                "XIII",
            ]
        ],
        winsize=2048,
        head_interval=16,
        transform=idx_to_onehot,
        target_transform=None,
    )
    valid_dataRAM = SequenceDatasetRAM(
        fasta_file,
        nuc_file,
        ["chrXIV", "chrXV"],
        winsize=2048,
        head_interval=16,
        train=False,
        transform=idx_to_onehot,
        target_transform=None,
    )
    train_dataloaderRAM = DataLoader(
        training_dataRAM,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_samplebalance,
    )
    valid_dataloaderRAM = DataLoader(
        valid_dataRAM,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
    )

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    model = BassenjiMnaseNetwork().to(device)

    loss_fn = torch_mae_cor
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloaderRAM, model, loss_fn, optimizer, max_batch_per_epoch=256)
        test(valid_dataloaderRAM, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pt")
    print("Saved PyTorch Model State to model.pt")
