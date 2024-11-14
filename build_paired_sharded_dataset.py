#!/usr/bin/env python

# This code takes an IP file of reads and a Control file of reads and creates
# a dataset separated into train, valid and test.
# The dataset in saved into an npz archive in the specified directory
#
# To execute this code run
# build_sharded_dataset.py -ip <IP file> -c <Control file> -out <directory>
# with other options available
#
# parameters :
# - IP file : npy file containing multiple read sequences from IP
# - Control file : npy file containing multiple read sequences from Control
# - directory : path of the directory to store the dataset files in

import argparse
import gzip
from itertools import repeat
from pathlib import Path

import numpy as np
import utils
from Bio import SeqIO


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    ip_file_pair : List[str]
        Paired-end fastq files containing ip reads
    ctrl_file_pair : List[str]
        Paired-end fastq files containing control reads
    output: str
        Path to the output directory and file name
    read_length : int
        Number of bases in reads. If unspecified, the read length is inferred
        from the maximum length in the first 100 sequences from each file. All
        reads will be truncated or extended with N values to fit this length.
    split_sizes : tuple[int], default=[2**23, 2**23]
        Number of test and valid samples in this order, remaining samples are
        train samples. Set value to 0 to ignore a split.
    shard_size : int, default=2**24
        Number of reads in a shard
    dN : bool, default=False
        If True, reads with N values are discarded, as well as reads shorter
        than `read_length`.
    """
    # Declaration of expexted arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip",
        "--ip_file_pair",
        help="IP paired fastq files",
        type=str,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--ctrl_file_pair",
        help="Control paired fastq files",
        type=str,
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "-out", "--out_dir", help="output dataset directory", type=str, required=True
    )
    parser.add_argument(
        "-rl",
        "--read_length",
        help="number of base pairs to encode reads on, optional",
        type=int,
    )
    parser.add_argument(
        "-split",
        "--split_sizes",
        help="numbers of reads to put in the test and valid sets, "
        "must be of length 2, set to 0 to ignore a split, default to "
        "2**22 each.",
        default=[2**23, 2**23],
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "-shard",
        "--shard_size",
        help="maximum number of reads in a shard, default to 2**24.",
        default=2**24,
        type=int,
    )
    parser.add_argument(
        "-dN",
        "--discardN",
        help="indicates to only use fully sized reads",
        action="store_true",
    )
    args = parser.parse_args()
    # Check if the input data is valid
    for file in args.ip_file_pair + args.ctrl_file_pair:
        if not Path(file).exists():
            raise ValueError(
                f"file {file} does not exist.\n" "Please enter valid fastq file paths."
            )
    return args


def parse_fastq(filename, discardN=False, read_length=None):
    """Generator for reading fastq and skipping unwanted reads.

    Supports regular and gzipped fastq.

    Parameters
    ----------
    filename : str
        Fastq file containing reads
    discardN : bool, optional
        If True, reads with Ns are skipped, as well as reads shorter
        than `read_length`.
    read_length : int, optional
        Length of reads in bases. If `discardN` is set to True, reads
        shorter than this length will be discarded.

    Yields
    ------
    read : Bio.Seq.SeqRecord
        Successive read records in fastq
    """

    def skip(read):
        """Function defining whether a read should be skipped"""
        return discardN and (
            "N" in read.seq
            or (read_length is not None and len(read.seq) != read_length)
        )

    # Use SeqIO to parse fastq, optionally reading from gzip file
    if filename.endswith("gz"):
        with gzip.open(filename, "rt") as f:
            for read in SeqIO.parse(f, format="fastq"):
                if skip(read):
                    continue
                yield read
    else:
        for read in SeqIO.parse(filename, "fastq"):
            if skip(read):
                continue
            yield read


def parse_paired_fastq(filenames, discardN=False, read_length=None):
    """Generator for reading fastq and skipping unwanted reads.

    Supports regular and gzipped fastq.

    Parameters
    ----------
    filenames : List[str]
        Paired fastq files containing reads
    discardN : bool, optional
        If True, pairs with a read with Ns are skipped,
        as well as those with a read shorter than `read_length`.
    read_length : int, optional
        Length of reads in bases. If `discardN` is set to True,
        pairs with a read shorter than this length will be discarded.

    Yields
    ------
    read : Bio.Seq.SeqRecord
        Successive read records in fastq
    """

    def skip(read):
        """Function defining whether a read should be skipped"""
        return discardN and (
            "N" in read.seq
            or (read_length is not None and len(read.seq) != read_length)
        )

    # Parse both files simultaneously
    for read1, read2 in zip(*(parse_fastq(filename) for filename in filenames)):
        if read1.id != read2.id:
            raise ValueError("Paired reads do not have same ids")
        if skip(read1) or skip(read2):
            continue
        yield read1, read2


def process_fastq_and_save(
    ip_file_pair,
    ctrl_file_pair,
    out_dir,
    shard_size=2**24,
    split_sizes=[2**23, 2**23],
    read_length=None,
    discardN=False,
):
    """
    Read multiple fastq files and convert them into a sharded numpy dataset.

    The reads in the fastq files are tokenized and stored in shards.
    Each shard is a binary archive containing two arrays ids and reads:
    ids is an array of string ids in the fastq file and reads are the
    corresponding tokenized paired reads sequences, with shape
    (Number of reads, 2, read_length).

    Parameters
    ----------
    ip_file_pair : List[str]
        Paired-end fastq files containing ip reads
    ctrl_file_pair : List[str]
        Paired-end fastq files containing control reads
    out_dir : str
        Name of the output dataset directory, must be empty
    shard_size : int, default=2**24
        Number of reads in a shard
    split_sizes : tuple[int], default=[2**23, 2**23]
        Number of test and valid samples in this order, remaining samples are
        train samples. Set value to 0 to ignore a split.
    read_length : int, default=None
        Number of bases in reads, if None, the read length is inferred from
        the maximum length in the first 100 sequences from each file. All
        reads will be truncated or extended with N values to fit this length.
    discardN : bool, default=False
        if True, reads with N values are discarded, as well as reads shorter
        than `read_length`.
    """

    # helper functions
    def save_shard(shard):
        """Tokenize reads of a shard and save to npz archive"""
        print(f"saving shard {cur_split}_{cur_shard}...")
        if not discardN:  # extend reads before tokenizing
            shard = [read + "N" * (read_length - len(read)) for read in shard]
        reads = np.stack(utils.ordinal_encoder(shard)).reshape(-1, 2, read_length)
        np.savez_compressed(
            Path(out_dir, f"{cur_split}_{cur_shard}"), ids=ids, reads=reads
        )

    def get_split_iterators():
        """
        Return shard sizes for each split, in iterators.

        This is a generator function, generating an iterator per split, then
        an infinite iterator
        """
        for split_size in split_sizes:
            q, mod = divmod(split_size, shard_size)
            shard_sizes = [shard_size] * q
            if mod:
                shard_sizes.append(mod)
            yield iter(shard_sizes)
        yield repeat(shard_size)

    # Build output directory if needed
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Infer read length from first 300 sequences in each file
    if read_length is None:
        print("read length is unspecified, inferring read length from files")
        read_length = 0
        for i, reads in enumerate(
            zip(*(parse_fastq(file) for file in ip_file_pair + ctrl_file_pair))
        ):
            if i > 100:
                break
            read_length = max(read_length, max(len(read.seq) for read in reads))
        print("read length is", read_length)

    # Handle train-valid-test splits
    splits = zip(["test", "valid", "train"], get_split_iterators())
    # Initialize first split and counters
    cur_split, cur_split_shards = next(splits)
    cur_shard_size = next(cur_split_shards)
    cur_shard = 0
    ids, shard = [], []

    # Read files
    for (ip1, ip2), (ctrl1, ctrl2) in zip(
        *(
            parse_paired_fastq(file_pair, discardN=discardN, read_length=read_length)
            for file_pair in (ip_file_pair, ctrl_file_pair)
        )
    ):
        ids.extend([ip1.id, ctrl1.id])
        shard.extend([str(read.seq) for read in (ip1, ip2, ctrl1, ctrl2)])
        # When shard is full, save it
        if len(ids) >= cur_shard_size:
            save_shard(shard)
            # Reinitialize
            ids, shard = [], []
            cur_shard += 1
            while True:
                try:
                    # Update next shard size
                    cur_shard_size = next(cur_split_shards)
                except StopIteration:
                    # Split is done, get to next split,
                    # loop again in case split is empty
                    cur_split, cur_split_shards = next(splits)
                    cur_shard = 0
                else:
                    # cur_shard_size was set successfully
                    break

    # Save last incomplete shard
    if len(ids) != 0:
        save_shard(shard)


if __name__ == "__main__":
    args = parsing()
    process_fastq_and_save(**vars(args))
