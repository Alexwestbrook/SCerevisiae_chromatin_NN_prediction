import argparse
import datetime
import itertools as it
import json
import socket
import time
from pathlib import Path

import models
import numpy as np
import pandas as pd
import predict_pytorch
import torch
from matplotlib import pyplot as plt


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
        "-o", "--output_dir", type=str, required=True, help="output directory"
    )
    parser.add_argument(
        "-m",
        "--model_file",
        type=str,
        required=True,
        help="trained model state file in pytorch format",
    )
    parser.add_argument(
        "-arch",
        "--architecture",
        help="Name of the model architecture",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-w",
        "--winsize",
        type=int,
        required=True,
        help="Size of the window in bp to use for prediction",
    )
    parser.add_argument(
        "-h_int",
        "--head_interval",
        help="Spacing between output head in case of mutliple outputs",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-nt",
        "--n_tracks",
        help="Number of tracks predicted by the model (default: %(default)s)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-track",
        "--track_index",
        help="Track index to optimize on (default: %(default)s)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-crop",
        "--head_crop",
        help="Number of heads to crop on left and right side of window (default: %(default)s)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-ord",
        "--one_hot_order",
        type=str,
        default="ACGT",
        help="order of the one-hot encoding for the model (default: %(default)s)",
    )
    parser.add_argument(
        "-kfile",
        "--kmer_file",
        type=str,
        help="file with kmer distribution to use for initializing sequences. "
        "The first named column is considered for the frequency values. "
        "There should be k columns before that, forming the kmers as a multi-index",
    )
    parser.add_argument(
        "-n",
        "--n_seqs",
        type=int,
        default=1,
        help="number of sequences to generate simultaneously (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=2175,
        help="length of the sequences to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-start",
        "--start_seqs",
        type=str,
        help="numpy binary file containing starting sequences as indexes with "
        "shape (n_seqs, length). If set overrides n_seqs and length.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of steps to perform (default: %(default)s)",
    )
    parser.add_argument(
        "-dist",
        "--distribute",
        action="store_true",
        help="indicates to use MirrorStrategy for prediction",
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        default=1,
        help="specifies a stride in predictions to go faster (default: %(default)s)",
    )
    parser.add_argument(
        "-mid",
        "--middle_pred",
        action="store_true",
        help="specifies to predict only on middle window",
    )
    parser.add_argument(
        "--flanks",
        type=str,
        help="file with flanking sequences to use for prediction, can also be "
        "'random' to get random flanks with specified kmer distribution "
        "or 'self' to use itself as flank as in tandem repeats",
    )
    parser.add_argument(
        "-targ",
        "--target",
        type=float,
        nargs="+",
        default=[0],
        help="target profile for the designed sequences. If a single value, "
        "consider a flat target for the entire sequence, otherwise must "
        "be of same length as the sequences. (default: %(default)s)",
    )
    parser.add_argument(
        "-targ_rev",
        "--target_rev",
        type=float,
        nargs="+",
        help="target profile for the designed sequences on the reverse "
        "strand, if unspecified, consider the same target for forward "
        "and reverse. If a single value, consider a flat target for the "
        "entire sequence, otherwise must be of same length as the "
        "sequences.",
    )
    parser.add_argument(
        "-amp",
        "--amplitude",
        type=float,
        default=1,
        help="amplitude for target profile (default: %(default)s)",
    )
    parser.add_argument(
        "-ilen", "--insertlen", type=int, help="length of insert in target"
    )
    parser.add_argument(
        "-ishape",
        "--insertshape",
        type=str,
        default="linear",
        help="shape of insert. Must be either 'block', 'deplete', "
        "'linear', 'sigmoid' or 'gaussian' (default: %(default)s)",
    )
    parser.add_argument(
        "-istart",
        "--insertstart",
        type=int,
        help="Index of the position where the insert should start",
    )
    parser.add_argument(
        "-bg",
        "--background",
        type=str,
        nargs=2,
        default=["low", "low"],
        help="background signal to put at the left and right of the insert (default: %(default)s)",
    )
    parser.add_argument(
        "-stdf",
        "--std_factor",
        type=float,
        default=1 / 4,
        help="standard deviation for gaussian peak as fraction of length (default: %(default)s)",
    )
    parser.add_argument(
        "-sigs",
        "--sig_spread",
        type=float,
        default=6,
        help="absolute value to compute the sigmoid up to on each side (default: %(default)s)",
    )
    parser.add_argument(
        "-per", "--period", type=int, help="period of the target, makes it multipeak"
    )
    parser.add_argument(
        "-plen", "--periodlen", type=int, help="length of periodic peak in target"
    )
    parser.add_argument(
        "-pshape",
        "--periodshape",
        type=str,
        default="gaussian",
        help="shape of periodic peak. Must be either 'block', 'deplete', "
        "'linear', 'sigmoid' or 'gaussian' (default: %(default)s)",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        help="numpy binary file containing the target profile for the "
        "designed sequences. If set, overrides target, target_rev and "
        "also length, unless start_seqs is set, in which case it must be "
        "of same length as start_seqs. Can also be an npz archive with 2 "
        "different targets for each strand, with keys named 'forward' "
        "and 'reverse'.",
    )
    parser.add_argument(
        "-targ_gc",
        "--target_gc",
        type=float,
        default=0.3834,
        help="target GC content for the designed sequences (default: %(default)s)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="rmse",
        help="loss to use for comparing prediction and target (default: %(default)s)",
    )
    parser.add_argument(
        "-wgt",
        "--weights",
        type=float,
        nargs=4,
        default=[1, 1, 1, 1],
        help="weights for the energy terms in order E_gc, E_for, E_rev, E_mut (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for kMC (default: %(default)s)",
    )
    parser.add_argument(
        "--mutfree_pos_file",
        type=str,
        help="Numpy binary file with positions where mutations can be performed.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        help="number of mutations to predict on at once (default: %(default)s)",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="number of parallel workers for preprocessing batches (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="seed to use for random generations (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="whether to print information messages",
    )
    args = parser.parse_args()
    # Basic checks
    assert len(args.one_hot_order) == 4 and set(args.one_hot_order) == set("ACGT")
    for item in [
        args.n_seqs,
        args.length,
        args.steps,
        args.stride,
        args.batch_size,
        args.period,
        args.periodlen,
        args.insertstart,
        args.n_tracks,
        args.winsize,
        args.head_interval,
        args.num_workers,
    ]:
        assert item is None or item >= 1
    assert isinstance(args.track_index, int) and args.track_index < args.n_tracks
    assert args.temperature > 0
    assert args.target_gc < 1 and args.target_gc > 0
    # Check starting sequences
    if args.start_seqs is not None:
        seqs = np.load(args.start_seqs)
        assert seqs.ndim == 2
        args.n_seqs, args.length = seqs.shape
    # Extract target profile
    if args.target_file is not None:
        with np.load(args.target_file) as f:
            if isinstance(f, np.ndarray):
                args.target = f
                args.target_rev = f
            else:
                args.target = f["forward"]
                args.target_rev = f["reverse"]
        if args.start_seqs is None:
            args.length = len(args.target)
    elif args.insertlen is not None:
        args.target = make_target(
            args.length,
            args.insertlen,
            args.amplitude,
            ishape=args.insertshape,
            insertstart=args.insertstart,
            background=args.background,
            period=args.period,
            pinsertlen=args.periodlen,
            pshape=args.periodshape,
            std_factor=args.std_factor,
            sig_spread=args.sig_spread,
        )
        args.target_rev = args.target
    else:
        # Forward target
        if len(args.target) == 1:
            args.target = np.full(args.length, args.target[0], dtype=float)
        else:
            args.target = np.array(args.target, dtype=float)
        # Reverse target
        if args.target_rev is None:
            args.target_rev = args.target
        elif len(args.target_rev) == 1:
            args.target_rev = np.full(args.length, args.target_rev[0], dtype=float)
        else:
            args.target_rev = np.array(args.target_rev, dtype=float)
    assert len(args.target) == len(args.target_rev) and len(args.target) == args.length
    return args


def GC_energy(seqs, target_gc):
    """Compute GC energy for each sequence in an array.

    The GC_energy is computed as `abs(gc - target_gc)`

    Parameters
    ----------
    seqs : ndarray
        Array of indexes into 'ACGT', with the sequences on the last axis
    target_gc : float
        Value of the target gc content

    Returns
    -------
    ndarray
        Array of absolute difference between sequence gc and target gc,
        with same shape as seqs but last dimension removed
    """
    return np.abs(
        np.sum((seqs == 1) | (seqs == 2), axis=-1) / seqs.shape[-1] - target_gc
    )


def all_mutations(seqs, seen_bases, mutfree_pos=None):
    """Perform all possible mutations and associate each its mutation energy.

    Mutation energy of a mutation X -> Y is defined as the number of times Y
    has been seen at this position during the optimization of this particular
    sequence.

    Parameters
    ----------
    seqs : ndarray, shape=(n, l)
        Array of indexes into 'ACGT', n is the number of sequences and l their
        length.
    seen_bases : ndarray, shape=(n, l, 4)
        Array of occurence of each base at each position during the
        optimization process of each sequence
    mutfree_pos : ndarray, default=None
        Array of positions in length where mutations can be taken

    Returns
    -------
    mutated : ndarray, shape=(n, 3*l, l)
        Array of all possible mutations of each sequence in seqs
    mut_energy : ndarray, shape=(n, 3*l)
        Array of mutation energy associated with each mutation

    Notes
    -----
    Mutations are performed in a circular rotation (0 -> 1 -> 2 -> 3 -> 0) by
    adding 1, 2 or 3 to the index and then taking the modulo 4.
    """
    n_seqs, length = seqs.shape
    # Create mutations array
    # Array of increments of 1, 2 or 3 at each position for a single sequence
    single_seq_increments = repeat_along_diag(
        np.arange(1, 4, dtype=seqs.dtype).reshape(-1, 1), length
    )  # shape (3*length, length)
    # Add increments to each sequence by broadcasting and take modulo 4
    mutated = (
        np.expand_dims(seqs, axis=1) + np.expand_dims(single_seq_increments, axis=0)
    ) % 4
    # Associate each mutation with its mutation energy
    # Array of resulting bases for each mutation
    mut_idx = (
        np.expand_dims(seqs, axis=-1) + np.arange(1, 4).reshape(1, -1)
    ) % 4  # shape (n_seqs, length, 3)
    # Select values associated with each base in seen_bases
    mut_energy = np.take_along_axis(seen_bases, mut_idx, axis=-1).reshape(
        n_seqs, 3 * length
    )
    # Keep mutations only at specific positions
    if mutfree_pos is not None:
        mutfree_pos = np.ravel(
            mutfree_pos.reshape(-1, 1) * 3 + np.arange(3).reshape(1, -1)
        )
        mutated = mutated[:, mutfree_pos]
        mut_energy = mut_energy[:, mutfree_pos]
    return mutated, mut_energy


def repeat_along_diag(a, r):
    """Construct a matrix by repeating a sub_matrix along the diagonal.

    References
    ----------
    https://stackoverflow.com/questions/33508322/create-block-diagonal-numpy-array-from-a-given-numpy-array
    """
    m, n = a.shape
    out = np.zeros((r, m, r, n), dtype=a.dtype)
    diag = np.einsum("ijik->ijk", out)
    diag[:] = a
    return out.reshape(-1, n * r)


def exp_normalize(x, axis=-1):
    res = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return res / np.sum(res, axis=axis, keepdims=True)


def random_sequences(
    n_seqs: int,
    seq_length: int,
    freq_kmers: pd.Series,
    out: str = "seq",
    rng: np.random.Generator = np.random.default_rng(),
) -> np.ndarray:
    """Generate random DNA sequences with custom kmer distribution.

    Parameters
    ----------
    n_seqs : int
        Number of sequences to generate
    seq_length : int
        Length of the sequences to generate, must be greater than k
    freq_kmers : Series
        Series containing frequencies or occurences of each k-mer.
        Must be indexed by a k-level MultiIndex with the bases 'ACGT' on each
        level in this order.
    out : {'seq', 'idx', 'one_hot'}
        Output format, 'seq' for nucleotide characters, 'idx' for indices into
        'ACGT' or 'one_hot' for one-hot encoded bases
    rng: np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ndarray, shape=(`n_seqs`, `seq_length`)
        Generated sequences as a 2D-array of characters, of indices into
        'ACGT' or 3D-array of one-hot encoded bases

    """
    assert n_seqs >= 1 and seq_length >= 0
    # Array of bases for fast indexing
    letters = np.array(list("ACGTN"))

    # Get value of k
    k = freq_kmers.index.nlevels
    if k == 1:
        seqs = rng.choice(4, size=(n_seqs, seq_length), p=freq_kmers)
    else:
        # Cumulative distribution of each base, given the previous k-1
        groups = freq_kmers.groupby(level=list(i for i in range(k - 1)))
        sum = groups.transform("sum")
        cumsum = groups.transform("cumsum")
        p_cum_kmers = cumsum / sum
        # Convert to kD-array
        arr_kmers = np.zeros(tuple([4] * k))
        for tup in it.product(range(4), repeat=k):
            arr_kmers[tup] = np.asarray(p_cum_kmers.loc[tuple(letters[i] for i in tup)])
        # Empty sequences
        seqs = np.array([4] * seq_length * n_seqs).reshape(n_seqs, seq_length)
        # Get first k-mer given k-mer distribution
        r_start = rng.choice(len(freq_kmers), n_seqs, p=freq_kmers / freq_kmers.sum())
        seqs[:, :k] = np.array(list(it.product(range(4), repeat=k)))[
            r_start, :seq_length
        ]
        # Generate random numbers for all iterations
        if seq_length > k:
            r = rng.random((n_seqs, seq_length - k))
        # Get other bases given k-mer distribution, previous (k-1)-mer and random
        # numbers
        for i in range(k, seq_length):
            seqs[:, i] = np.argmax(
                arr_kmers[
                    tuple(
                        arr.ravel()
                        for arr in np.split(seqs[:, i - k + 1 : i], k - 1, axis=1)
                    )
                ]
                >= r[:, [i - k] * 4],
                axis=1,
            )
    if out == "idx":
        return np.asarray(seqs, dtype=np.int8)
    elif out == "seq":
        return letters[seqs]
    elif out == "one_hot":
        return np.eye(4, dtype=bool)[seqs]


def get_profile_torch(
    seqs,
    model,
    winsize,
    head_interval=None,
    head_crop=0,
    n_tracks=1,
    track_index=0,
    middle=False,
    reverse=False,
    stride=1,
    batch_size=1024,
    num_workers=4,
    offset=None,
    seed=None,
    verbose=False,
    return_index=False,
    flanks=None,
):
    """Predict profile.

    Parameters
    ----------
    seqs : ndarray
        Array of indexes into 'ACGT', sequences to predict on are read on the
        last axis
    model
        Tensorflow model instance supporting the predict method
    winsize : int
        Input length of the model
    head_interval : int, default=None
        Spacing between outputs of the model, for a model with multiple
        outputs starting at first window position and with regular spacing.
        If None, model must have a single output in the middle of the window.
    head_crop : int, optional
        Number heads to not consider at left and right of the prediction window
    n_tracks : int, optional
        Number of tracks outputted by the model
    trackindex : int, optional
        Track index to predict on
    middle : bool, default=False
        Whether to use only the middle half of output heads for deriving
        predictions. This results in no predictions on sequence edges.
    reverse : bool, default=False
        If True predict on reverse strand. Default is False for predicting on
        forward strand.
    stride : int, default=1
        Stride to use for prediction. Using a value other than 1 will result
        in bases being skipped and make prediction faster
    batch_size : int, default=1024
        Batch_size for model.predict().
    num_worker : int, optional
        Number of processes to run in parrallel when making batch for prediction
    offset : int, default=None
        Offset for start of prediciton, will be forced to be positive and
        smaller than stride by taking the modulo. Default value of None will
        result in a random offset being chosen.
    seed : int, default=None
        Value of seed to use for choosing random offset.
    verbose : bool, default=False
        If True, print information messages.
    return_index : bool, default=False
        If True, return indices corresponding to the predictions.
    flanks : tuple of array or 'self', default=None
        Tuple of 2 arrays flank_keft and flank_right to be added at the start
        and end respectively of each sequence to get prediction on the entire
        sequence.

    Returns
    -------
    preds : ndarray
        Array of predictions with same shape as seqs, except on the last
        dimension, containing predictions for that sequence.
    indices : ndarray
        Array of indices of preds into seqs to be taken with
        np.take_along_axis, only provided if return_index is True.
    """
    # Copy sequence array and make 2D with sequence along the 2nd axis
    seqs2D = seqs.reshape(-1, seqs.shape[-1]).copy()
    # Determine offset for prediction
    if offset is None:
        # Randomize offset
        if seed is not None:
            np.random.seed(seed)
        offset = np.random.randint(0, stride)
    else:
        offset %= stride
    if verbose:
        print(f"Predicting with stride {stride} and offset {offset}")
    # pred_start: distance between first prediction and sequence start
    # pred_stop: distance between last prediction and sequence end
    pred_start = 0
    pred_stop = head_interval - 1
    jump_stride = None
    kept_heads_start = None
    # Increase distances in case of middle predictions
    if middle:
        pred_start += winsize // 4
        pred_stop += winsize // 4
        jump_stride = winsize // 2
        kept_heads_start = winsize // (4 * head_interval) - head_crop
    # Add flanking sequences to make prediction along the entire sequence, and
    # update distances
    if flanks is not None:
        if reverse:
            leftpad, rightpad = pred_stop, pred_start
        else:
            leftpad, rightpad = pred_start, pred_stop
        if flanks == "self":
            flank_left = np.tile(seqs2D, leftpad // seqs2D.shape[-1] + 1)
            flank_left = flank_left[:, flank_left.shape[-1] - leftpad :]
            flank_right = np.tile(seqs2D, rightpad // seqs2D.shape[-1] + 1)
            flank_right = flank_right[:, :rightpad]
            seqs2D = np.hstack([flank_left, seqs2D, flank_right])
        else:
            flank_left, flank_right = flanks
            flank_left = flank_left[len(flank_left) - leftpad :]
            flank_right = flank_right[:rightpad]
            seqs2D = np.hstack(
                [
                    np.tile(flank_left, (len(seqs2D), 1)),
                    seqs2D,
                    np.tile(flank_right, (len(seqs2D), 1)),
                ]
            )
    # Make predictions
    preds, indices = predict_pytorch.predict(
        model,
        seqs2D,
        winsize,
        head_interval,
        head_crop=head_crop,
        n_tracks=n_tracks,
        reverse=reverse,
        stride=stride,
        offset=offset,
        jump_stride=jump_stride,
        kept_heads_start=kept_heads_start,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
    )
    # Select track and reshape as original sequence
    preds = preds[..., track_index].reshape(seqs.shape[:-1] + (-1,))
    if return_index:
        if flanks is not None:
            indices -= flank_left.shape[-1]
        return preds, indices
    else:
        return preds


def rmse(y_true, y_pred):
    """Compute RMSE between two arrays.

    Parameters
    ----------
    y_true, y_pred : ndarray
        Arrays of values. If multidimensional, computation is performed along
        the last axis.

    Returns
    -------
    ndarray
        Array of same shape as y_true and y_pred but with last axis removed
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=-1))


def np_mae_cor(y_true, y_pred, axis=-1):
    """Numpy equivalent of mae_cor"""
    X = y_true - np.mean(y_true)
    Y = y_pred - np.mean(y_pred, axis=axis, keepdims=True)

    sigma_XY = np.sum(X * Y, axis=axis)
    sigma_X = np.sqrt(np.sum(X * X))
    sigma_Y = np.sqrt(np.sum(Y * Y, axis=axis))

    cor = sigma_XY / (sigma_X * sigma_Y + np.finfo(y_pred.dtype).eps)
    mae = np.mean(np.abs(y_true - y_pred), axis=axis)

    return (1 - cor) + mae


def make_shape(
    length,
    amplitude,
    shape="linear",
    background=["low", "low"],
    std_factor=1 / 4,
    sig_spread=6,
):
    base_length = length
    if background[0] == background[1]:
        length = (length + 1) // 2
    if shape == "block":
        arr = np.full(length, amplitude)
    elif shape == "deplete":
        arr = np.zeros(length)
    elif shape == "linear":
        arr = np.linspace(0, amplitude, length)
    elif shape == "gaussian":
        if background[0] != background[1]:
            base_length = length * 2 + 1
        arr = np.exp(
            -((np.arange(length) - (base_length - 1) / 2) ** 2)
            / (2 * (base_length * std_factor) ** 2)
        )
        arr -= np.min(arr)
        arr *= amplitude / np.max(arr)
    elif shape == "sigmoid":
        x = np.linspace(-sig_spread, sig_spread, length)
        arr = 1 / (1 + np.exp(-x))
        arr -= np.min(arr)
        arr *= amplitude / np.max(arr)
    if background[0] == "high":
        arr = amplitude - arr
    if background[0] == background[1]:
        if base_length % 2 != 0:
            return np.concatenate([arr, arr[-2::-1]])
        else:
            return np.concatenate([arr, arr[::-1]])
    else:
        return arr


def make_target(
    length,
    insertlen,
    amplitude,
    ishape="linear",
    insertstart=None,
    period=None,
    pinsertlen=147,
    pshape="gaussian",
    background=["low", "low"],
    **kwargs,
):
    if insertstart is None:
        insertstart = (length - insertlen + 1) // 2
    rightlen = length - insertlen - insertstart
    if period is not None:
        if insertlen == 0:
            insertlen = period - pinsertlen
            insertstart = (length - insertlen + 1) // 2
            rightlen = length - insertlen - insertstart
        insert = make_shape(insertlen, amplitude, shape="deplete")
        pinsert = make_shape(pinsertlen, amplitude, shape=pshape, **kwargs)
        tile = np.concatenate([pinsert, np.zeros(period - len(pinsert))])
        backgr = np.tile(tile, length // (period) + 1)
        leftside = backgr[insertstart - 1 :: -1]
        rightside = backgr[:rightlen]
    else:
        insert = make_shape(
            insertlen, amplitude, shape=ishape, background=background, **kwargs
        )
        backgr_dict = {
            "low": lambda x: np.zeros(x),
            "high": lambda x: np.full(x, amplitude),
        }
        leftside = backgr_dict[background[0]](insertstart)
        rightside = backgr_dict[background[1]](rightlen)
    target = np.concatenate([leftside, insert, rightside])
    return target[:length]


def select(energies, weights, cur_energy, temperature, step=None):
    """Choose a mutation with low energy based on kMC method.

    Parameters
    ----------
    energies : list of arrays, len=n_energies
        List of energy components of the total energy. Energy components must
        be of shape (n_seqs, n_mutations)
    weights : array_like, len=n_energies
        Value of weights for each energy component.
    cur_energy : ndarray, shape=(n_seqs)
        Energies of the sequences before mutation.
    temperature : float
        Temperature to use for deriving probabilities from energies in the kMC
    step : int, default=None
        Step index in the optimisation process. If set, the computed
        probabilities will be save to a file.

    Returns
    -------
    sel_idx : ndarray, shape=(n_seqs)
        Array of indices of selected mutation for each sequence
    sel_energy : ndarray, shape=(n_seqs, n_energies+1)
        Array of energies associated to selected mutations for each sequence.
        First is total energy, followed by each component.

    Notes
    -----
    The kinetic Monte-Carlo (kMC) method consists in deriving energy values
    for all options and selecting a low energy option in a probabilistic way
    by computing exp(-E_i/T) / sum_i(exp(-E_i/T)) where T is the temperature.
    """
    # Compute energy and probability
    tot_energy = sum(w * e for w, e in zip(weights, energies))
    old_prob = exp_normalize(-tot_energy / temperature)
    prob = exp_normalize((cur_energy.reshape(-1, 1) - tot_energy) / temperature)
    # Maybe save probabilities
    if step is not None:
        np.save(Path(args.output_dir, "probs", f"old_prob_step{step}.npy"), old_prob)
        np.save(Path(args.output_dir, "probs", f"prob_step{step}.npy"), prob)
    # Select by the position of a random number in the cumulative sum
    cumprob = np.cumsum(prob, axis=-1)
    r = np.random.rand(len(prob), 1)
    sel_idx = np.argmax(r <= cumprob, axis=-1)
    # Associate energy to selected sequences
    sel_energies = np.stack(
        [en[np.arange(len(en)), sel_idx] for en in [tot_energy] + energies], axis=1
    )
    return sel_idx, sel_energies


def get_rows_and_cols(n_seqs):
    """Determine number of rows and columns to place a number of subplots.

    Parameters
    ----------
    n_seqs : int
        Number of subplots to place

    Returns
    -------
    nrow, ncol
        Number of rows and columns to use
    """
    if n_seqs > 25:
        nrow, ncol = 5, 5  # plot max 25 sequences
    else:
        ncol = 5  # max number of columns
        while n_seqs % ncol != 0:
            ncol -= 1
        nrow = n_seqs // ncol
    return nrow, ncol


def main(args):  # TODO
    # Load model
    args.architecture = models.ARCHITECTURES[args.architecture]
    model = args.architecture(args.n_tracks).to(args.device)
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    # Tune predicting functions according to args
    assert args.head_interval % args.stride == 0

    def predicter(seqs, reverse=False, offset=None, flanks=None):
        return get_profile_torch(
            seqs,
            model,
            args.winsize,
            args.head_interval,
            head_crop=args.head_crop,
            n_tracks=args.n_tracks,
            track_index=args.track_index,
            middle=args.middle_pred,
            reverse=reverse,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            offset=offset,
            verbose=args.verbose,
            return_index=True,
            flanks=flanks,
        )

    # Extract flanking sequences
    if args.flanks in ["random", "self"]:
        if args.head_interval is not None:
            pad = args.head_interval - 1
            # Increase distances in case of middle predictions
            if args.middle_pred:
                pad += args.winsize // 4
        else:
            pad = args.winsize // 2
        flanks = args.flanks
    elif args.flanks is not None:
        with np.load(args.flanks) as f:
            flank_left = f["left"]
            flank_right = f["right"]
            assert flank_left.ndim == flank_right.ndim
            if flank_left.ndim == 2:
                assert len(flank_left) == len(flank_right)
                flanks = "choose_idx"
            else:
                assert flank_left.ndim == 1
                flanks = (flank_left, flank_right)
    else:
        flanks = None
    # Extract positions where mutations can be performed
    if args.mutfree_pos_file is not None:
        mutfree_pos = np.load(args.mutfree_pos_file)
    else:
        mutfree_pos = None
    # Extract kmer distribution
    if args.kmer_file is None:
        # Use target gc
        freq_kmer = pd.Series(
            [
                (1 - args.target_gc) / 2,
                args.target_gc / 2,
                args.target_gc / 2,
                (1 - args.target_gc) / 2,
            ],
            index=list("ACGT"),
        )
    else:
        with open(args.kmer_file) as f:
            for k, c in enumerate(f.readline()):
                if c != ",":
                    break
        freq_kmer = pd.read_csv(args.kmer_file, index_col=np.arange(k)).iloc[:, 0]
    # Generate and save start sequences
    if args.seed != -1:
        np.random.seed(args.seed)
    if args.start_seqs:
        seqs = np.load(args.start_seqs)
    else:
        seqs = random_sequences(args.n_seqs, args.length, freq_kmer, out="idx")
    np.save(Path(args.output_dir, "designed_seqs", "start_seqs.npy"), seqs)
    # Compute energy of start sequences
    # Predict on forward and reverse strands
    if flanks == "random":
        randseqs = random_sequences(2, pad, freq_kmer, out="idx")
        flanks = (randseqs[0], randseqs[1])
    elif flanks == "choose_idx":
        flank_idx = np.random.randint(0, len(flank_left))
        flanks = (flank_left[flank_idx], flank_right[flank_idx])
        if args.verbose:
            print(f"Using flank_idx {flank_idx}")
    preds, indices = predicter(
        seqs, offset=np.random.randint(0, args.stride), flanks=flanks
    )
    preds_rev, indices_rev = predicter(
        seqs, offset=np.random.randint(0, args.stride), flanks=flanks, reverse=True
    )
    # Compute energy
    gc_energy = GC_energy(seqs, args.target_gc)
    for_energy = args.loss(args.target[indices], preds)
    rev_energy = args.loss(args.target_rev[indices_rev], preds_rev)
    energy_list = [gc_energy, for_energy, rev_energy, np.zeros(args.n_seqs)]
    cur_energy = sum(w * e for w, e in zip(args.weights, energy_list))
    with open(Path(args.output_dir, "energy.txt"), "a") as f:
        np.savetxt(
            f,
            np.stack([cur_energy] + energy_list, axis=1),
            fmt="%-8e",
            delimiter="\t",
            header="start",
        )
    # Initialize array of already seen bases for each position
    seen_bases = np.eye(4, dtype=int)[seqs]
    # Determine figure parameters
    nrow, ncol = get_rows_and_cols(args.n_seqs)
    target_by_strand = not np.all(args.target == args.target_rev)
    for step in range(args.steps):
        if args.verbose:
            print(time.time() - t0)
            print(f"Step {step}")
        # Generate all mutations, and associated mutation energy
        seqs, mut_energy = all_mutations(seqs, seen_bases, mutfree_pos)
        # Predict on forward and reverse strands
        if flanks == "random":
            randseqs = random_sequences(2, pad, freq_kmer, out="idx")
            flanks = (randseqs[0], randseqs[1])
        elif flanks == "choose_idx":
            flank_idx = np.random.randint(0, len(flank_left))
            flanks = (flank_left[flank_idx], flank_right[flank_idx])
            if args.verbose:
                print(f"Using flank_idx {flank_idx}")
        preds, indices = predicter(
            seqs, offset=np.random.randint(0, args.stride), flanks=flanks
        )
        preds_rev, indices_rev = predicter(
            seqs, offset=np.random.randint(0, args.stride), flanks=flanks, reverse=True
        )
        # Compute energy components
        gc_energy = GC_energy(seqs, args.target_gc)
        for_energy = args.loss(args.target[indices], preds)
        rev_energy = args.loss(args.target_rev[indices_rev], preds_rev)
        energy_list = [gc_energy, for_energy, rev_energy, mut_energy]
        # Choose best mutation by kMC method
        sel_idx, sel_energies = select(
            energy_list, args.weights, cur_energy, args.temperature, step=step
        )
        cur_energy = sel_energies[:, 0]
        # Keep selected sequence and increment seen_bases
        seqs = seqs[np.arange(len(seqs)), sel_idx]
        seen_bases[
            np.arange(len(seqs)), sel_idx // 3, seqs[np.arange(len(seqs)), sel_idx // 3]
        ] += 1
        # Save sequence, energy and plot profile
        np.save(
            Path(args.output_dir, "designed_seqs", f"mut_seqs_step{step}.npy"), seqs
        )
        with open(Path(args.output_dir, "energy.txt"), "a") as f:
            np.savetxt(
                f, sel_energies, fmt="%-8e", delimiter="\t", header=f"step{step}"
            )
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(2 + 3 * ncol, 1 + 2 * nrow),
            facecolor="w",
            layout="tight",
            sharey=True,
        )
        if args.n_seqs == 1:
            ax_list = [axes]
        else:
            ax_list = axes.flatten()
        for ax, pfor, prev in zip(
            ax_list,
            preds[np.arange(len(seqs)), sel_idx],
            preds_rev[np.arange(len(seqs)), sel_idx],
        ):
            ax.plot(args.target, color="k", label="target")
            if target_by_strand:
                ax.plot(-args.target_rev, color="k")
                prev = -prev
            ax.plot(indices, pfor, label="forward")
            ax.plot(indices_rev, prev, label="reverse", alpha=0.8)
            ax.legend()
        fig.savefig(
            Path(args.output_dir, "pred_figs", f"mut_preds_step{step}.png"),
            bbox_inches="tight",
        )
        plt.close()
    if args.verbose:
        print(time.time() - t0)


if __name__ == "__main__":
    tmstmp = datetime.datetime.now()
    t0 = time.time()
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
    # Build output directory and initialize energy file
    Path(args.output_dir).mkdir(parents=True, exist_ok=False)
    Path(args.output_dir, "pred_figs").mkdir()
    Path(args.output_dir, "designed_seqs").mkdir()
    Path(args.output_dir, "probs").mkdir()
    with open(Path(args.output_dir, "energy.txt"), "w") as f:
        f.write(
            "# total_energy\t"
            "gc_energy\t"
            "for_energy\t"
            "rev_energy\t"
            "mut_energy\n"
        )
    # Store arguments in config file
    to_serealize = {
        k: v
        for k, v in vars(args).items()
        if k not in ["target", "target_rev", "weights"]
    }
    with open(Path(args.output_dir, "config.txt"), "w") as f:
        json.dump(to_serealize, f, indent=4)
        f.write("\n")
        f.write(f"weights: {args.weights}\n")
        f.write(f"timestamp: {tmstmp}\n")
        f.write(f"machine: {socket.gethostname()}\n")
    # Convert to non json serializable objects
    losses = {"rmse": rmse, "mae_cor": np_mae_cor}
    args.loss = losses[args.loss]
    args.weights = np.array(args.weights, dtype=float)
    # Save target
    np.savez(
        Path(args.output_dir, "target.npz"),
        forward=args.target,
        reverse=args.target_rev,
    )
    # Start computations, save total time even if there was a failure
    try:
        main(args)
    except KeyboardInterrupt:
        with open(Path(args.output_dir, "config.txt"), "a") as f:
            f.write("KeyboardInterrupt\n")
            f.write(f"total time: {time.time() - t0}\n")
        raise
    with open(Path(args.output_dir, "config.txt"), "a") as f:
        f.write(f"total time: {time.time() - t0}\n")
