#!/usr/bin/env python
import collections
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import Bio
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


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


def make_iterator(obj: Any, length: int = 1) -> Iterator:
    if isinstance(obj, collections.abc.Iterable):
        return iter(obj)
    else:
        return (obj for _ in range(length))


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
    axis: None or int or iterable of ints, optional
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


def apply_on_index(
    func: Callable,
    *args: Tuple[np.ndarray, np.ndarray],
    length: int = None,
    neutral: int = 0,
) -> np.ndarray:
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
    length : int, optional
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.
    neutral : int, optional
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


def mean_on_index(*args: Tuple[np.ndarray, np.ndarray], length: int = None):
    """Computes the mean of arrays specified as indices and values.

    Parameters
    ----------
    args : tuple of tuple of arrays
        Each argument represents an array, it must be a tuple
        (indices, values) for each array. Indices must be 1D, values can be
        multi-dimensional, in which case indices will be taken along the last
        axis.
    length : int, optional
        Length of the output array. If None, the smallest possible length is
        inferred from the indices in args.

    Returns
    -------
    ndarray
        Result array of the full length, including nans where none of the
        arrays had any values.
    """
    return apply_on_index(
        lambda n, *args: np.divide(sum(args), n, where=(n != 0)), *args, length=length
    )


def safe_filename(file: Union[Path, str]) -> Path:
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


def sliding_GC(
    seqs: np.ndarray, n: int, axis: int = -1, order: str = "ACGT", form: str = "token"
) -> np.ndarray:
    """Compute sliding GC content on encoded DNA sequences

    Sequences can be either tokenized or one-hot encoded.
    GC content will be computed by considering only valid tokens or one-hots.
    Valid tokens are between in the range [0, 4[, and valid one_hots have exactly one value equal to 1. However we check only if they sum to one.

    Parameters
    ----------
    seqs: ndarray
        Input sequences. Sequences are assumed to be read along last axis, otherwise change `axis` parameter.
    n: int
        Length of window to compute GC content on, must be greater than 0.
    axis: int, optional
        Axis along which to compute GC content. The default (-1) assumes sequences are read along last axis.
        Currently, form 'one_hot' doesn't support the axis parameter, it assumes one-hot values on last axis, and sequence on second to last axis.
    order: str, optional
        Order of encoding, must contain each letter of ACGT exactly once.
        If `form` is 'token', then value i corresponds to base at index i in order.
        If `form` is 'one_hot', then vector of zeros with a 1 at position i corresponds to base at index i in order.
    form: {'token', 'one_hot'}, optional
        Form of input array. 'token' for indexes of bases and 'one_hot' for one-hot encoded sequences, with an extra dimension.

    Returns
    -------
    ndarray
        Array of sliding GC content, with size along `axis` reduced by `n`-1, and optional one-hot dimension removed.
    """
    if form == "token":
        valid_mask = (seqs >= 0) & (seqs < 4)
        GC_mask = (seqs == order.find("C")) | (seqs == order.find("G"))
        if n > seqs.shape[axis]:
            n = seqs.shape[axis]
        return moving_sum(GC_mask, n, axis=axis) / moving_sum(valid_mask, n, axis=axis)
    elif form == "one_hot":
        valid_mask = seqs.sum(axis=-1) != 0
        GC_mask = seqs[:, [order.find("C"), order.find("G")]].sum(axis=-1)
        if n > seqs.shape[-1]:
            n = seqs.shape[-1]
        return moving_sum(GC_mask, n=n, axis=-1) / moving_sum(valid_mask, n=n, axis=-1)
    else:
        raise ValueError(f"form must be 'token' or 'one_hot', not {form}")
