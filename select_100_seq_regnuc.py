import json
from pathlib import Path

import kMC_sequence_design_pytorch as kmc
import models
import numpy as np
import torch
import utils
from kMC_sequence_design_pytorch import get_profile_torch


def make_peaks(
    peak_mask: np.ndarray, length_thres: int = 1, tol: int = 1
) -> np.ndarray:
    """Format peak array from peak boolean mask.

    Determine regions of consecutive high prediction, called peaks.

    Parameters
    ----------
    peak_mask : ndarray
        1D-array of boolean values along the chromosome
    length_thres : int, default=1
        Minimum length required for peaks, any peak strictly below that
        length will be discarded
    tol : int, default=1
        Distance between consecutive peaks under which the peaks are merged
        into one. Can be set higher to get a single peak when signal is
        fluctuating too much. Unlike slices, peaks include their end points,
        meaning [1 2] and [4 5] actually contain a gap of one base,
        but the distance is 2 (4-2). The default value of 1 means that no
        peaks will be merged.

    Returns
    -------
    peaks : ndarray, shape=(n, 2)
        2D-array, each line corresponds to a peak. A peak is a 1D-array of
        size 2, with format [peak_start, peak_end]. `peak_start` and
        `peak_end` are indices on the chromosome.
    """
    # Find where peak start and end
    change_idx = np.where(peak_mask[1:] != peak_mask[:-1])[0] + 1
    if peak_mask[0]:
        # If predictions start with a peak, add an index at the start
        change_idx = np.insert(change_idx, 0, 0)
    if peak_mask[-1]:
        # If predictions end with a peak, add an index at the end
        change_idx = np.append(change_idx, len(peak_mask))
    # # Check that change_idx contains as many starts as ends
    # assert (len(change_idx) % 2 == 0)
    # Merge consecutive peaks if their distance is below a threshold
    if tol > 1:
        # Compute difference between end of peak and start of next one
        diffs = change_idx[2::2] - change_idx[1:-1:2]
        # Get index when difference is below threshold, see below for matching
        # index in diffs and in change_idx
        # diff index:   0   1   2  ...     n-1
        # change index:1-2 3-4 5-6 ... (2n-1)-2n
        (small_diff_idx,) = np.where(diffs <= tol)
        delete_idx = np.concatenate((small_diff_idx * 2 + 1, small_diff_idx * 2 + 2))
        # Remove close ends and starts using boolean mask
        mask = np.ones(len(change_idx), dtype=bool)
        mask[delete_idx] = False
        change_idx = change_idx[mask]
    # Reshape as starts and ends
    peaks = np.reshape(change_idx, (-1, 2))
    # Compute lengths of peaks and remove the ones below given threshold
    lengths = np.diff(peaks, axis=1).ravel()
    peaks = peaks[lengths >= length_thres]
    return peaks


def predict_merge_nuc(
    seqs,
    flank_file="/home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz",
):
    # Get flanks
    with np.load(flank_file) as f:
        flank_left = f["left"]
        flank_right = f["right"]
    flanks = (flank_left, flank_right)
    # Make preds
    preds_nuc, indices_nuc = get_profile_torch(
        seqs, model_nuc, 2048, 16, middle=True, return_index=True, flanks=flanks
    )
    preds_nuc_rev, indices_nuc_rev = get_profile_torch(
        seqs,
        model_nuc,
        2048,
        16,
        reverse=True,
        middle=True,
        return_index=True,
        flanks=flanks,
    )
    # Merge forward and reverse
    length = seqs.shape[-1]
    return utils.mean_on_index(
        (indices_nuc, preds_nuc), (indices_nuc_rev, preds_nuc_rev), length=length
    )


def score_regnuc_2kb(
    preds,
    target_file="/home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/target.npz",
):
    # Get target
    with np.load(target_file) as f:
        target = f["forward"]
    return kmc.rmse(preds, target)


def predict_exp_regnuc(
    exp_name,
    steps=np.array([0, 20, 250, 500]),
    flank_file="/home/alex/shared_folder/SCerevisiae/data/S288c_siteManon_Int2_1kbflanks_ACGTidx.npz",
):
    # Get sequences
    seqs = []
    for step in steps:
        if step == 0:
            seqs.append(
                np.load(
                    f"/home/alex/SCerevisiae_chromatin_NN_prediction/generated/{exp_name}/designed_seqs/start_seqs.npy"
                )
            )
        else:
            seqs.append(
                np.load(
                    f"/home/alex/SCerevisiae_chromatin_NN_prediction/generated/{exp_name}/designed_seqs/mut_seqs_step{step-1}.npy"
                )
            )
    seqs = np.stack(seqs, axis=1)

    # Predict
    preds = predict_merge_nuc(seqs, flank_file)

    return preds


def count_NDRs(
    preds,
    NDR_thres=0.2,
    NDR_minlength=150,
):
    # Count NDRs
    NDRs = [[] for i in range(preds.shape[1])]
    for seq_idx in range(preds.shape[0]):
        for step_idx in range(preds.shape[1]):
            peaks = make_peaks(preds[seq_idx, step_idx] < NDR_thres)
            NDR = np.sum(np.diff(peaks, axis=1) > NDR_minlength)
            NDRs[step_idx].append(NDR)
    return np.array(NDRs)


if __name__ == "__main__":
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load model
    model_nuc = models.BassenjiMultiNetwork2(n_tracks=1).to(device)
    model_nuc_file = "Trainedmodels/model_myco_nuc_pt8/model_state.pt"
    model_nuc.load_state_dict(torch.load(model_nuc_file))
    model_nuc.eval()

    # Predict, score and count NDRS on all regnuc 2kb experiments:
    # "regnuc_2kb_1seq_randomflanks"
    # "regnuc_2kb_10seq_randomflanks"
    # "regnuc_2kb_10seq_flanksInt2"
    # "regnuc_2kb_100seq_randomflanks"
    all_score, all_preds, all_NDRs = [], [], []
    for exp in Path("generated").glob("regnuc_2kb*"):
        preds = predict_exp_regnuc(exp.name, steps=[500])
        score = score_regnuc_2kb(preds)
        NDRs = count_NDRs(preds, NDR_thres=0.5)
        all_preds.append(preds.squeeze(axis=1))
        all_score.append(score.squeeze(axis=1))
        all_NDRs.append(NDRs.squeeze(axis=0))
    score = np.concatenate(all_score)
    preds = np.concatenate(all_preds)
    NDRs = np.concatenate(all_NDRs)

    # Sort by score for selection, and keep 100 best sequences without NDRs
    sort_idx = score.argsort(axis=0)
    mask = np.ones(len(preds), dtype=bool)
    mask[NDRs[sort_idx] > 0] = False
    mask[104:] = False
    selected_preds = preds[sort_idx[mask]]
    selected_score = score[sort_idx[mask]]

    # Save origin of selected sequence as indexes in each experiment list of sequences
    seps = [0, 1, 11, 111, 121]
    sorted_selection = np.sort(sort_idx[mask])
    selected_seqs_idx = {}
    for i, exp in enumerate(Path("generated").glob("*regnuc_2kb*")):
        selected_seqs_idx[exp.name] = (
            sorted_selection[
                (sorted_selection >= seps[i]) & (sorted_selection < seps[i + 1])
            ]
            - seps[i]
        ).tolist()
    with open(
        "/home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs_origin.json",
        "w",
    ) as f:
        json.dump(selected_seqs_idx, f, indent=4)

    # Save selected sequences in a single file
    selected_seqs = []
    for exp_name, idx_list in selected_seqs_idx.items():
        seqs = np.load(
            f"/home/alex/SCerevisiae_chromatin_NN_prediction/generated/{exp_name}/designed_seqs/mut_seqs_step499.npy"
        )
        selected_seqs.append(seqs[np.array(idx_list)])
    np.save(
        "/home/alex/SCerevisiae_chromatin_NN_prediction/generated/regnuc_2kb_100seq_randomflanks/100_selected_seqs.npy",
        np.concatenate(selected_seqs),
    )
