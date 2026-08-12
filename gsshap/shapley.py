#!/usr/bin/env python
# coding: utf-8
"""Core Shapley and MMD utilities for GS-SHAP.

The Shapley implementation uses feature-wise mean replacement as the
baseline and activates group-segment player blocks along each permutation.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Callable, Optional

import numpy as np
import torch


# -----------------------
# RBF kernel & MMD (shared)
# -----------------------
def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> np.ndarray:
    """Return an RBF kernel matrix using the median heuristic when needed."""
    if gamma is None:
        Z = np.concatenate([X, Y], axis=0)  # (n + m, d)
        sq_dists = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
        med = np.median(sq_dists)
        gamma = 1.0 if med <= 0.0 else 1.0 / (2.0 * med)

    XX = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    YY = np.sum(Y**2, axis=1, keepdims=True)  # (m, 1)
    XY = X @ Y.T                              # (n, m)

    dists = XX - 2.0 * XY + YY.T              # (n, m)
    dists = np.maximum(dists, 0.0)

    exponent = -gamma * dists
    exponent = np.clip(exponent, -50.0, 0.0)

    return np.exp(exponent)


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute unbiased MMD^2 with an RBF kernel."""
    n = X.shape[0]
    m = Y.shape[0]
    if n < 2 or m < 2:
        return 0.0

    Kxx = rbf_kernel(X, X)
    Kyy = rbf_kernel(Y, Y)
    Kxy = rbf_kernel(X, Y)

    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    term_xx = Kxx.sum() / (n * (n - 1))
    term_yy = Kyy.sum() / (m * (m - 1))
    term_xy = 2.0 * Kxy.sum() / (n * m)
    return float(term_xx + term_yy - term_xy)


def _resolve_torch_device(device: Optional[torch.device | str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _rbf_kernel_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    gamma: float | None = None,
) -> torch.Tensor:
    """Torch implementation of the RBF kernel used by segmentation."""
    X = X.float()
    Y = Y.float()

    if gamma is None:
        Z = torch.cat([X, Y], dim=0)
        sq_dists = torch.cdist(Z, Z, p=2.0).pow(2)
        med = torch.median(sq_dists)
        med_val = float(med.item())
        gamma = 1.0 if med_val <= 0.0 else 1.0 / (2.0 * med_val)

    XX = (X ** 2).sum(dim=1, keepdim=True)
    YY = (Y ** 2).sum(dim=1, keepdim=True)
    XY = X @ Y.transpose(0, 1)

    dists = XX - 2.0 * XY + YY.transpose(0, 1)
    dists = torch.clamp(dists, min=0.0)

    exponent = -float(gamma) * dists
    exponent = torch.clamp(exponent, min=-50.0, max=0.0)
    return torch.exp(exponent)


def mmd2_unbiased_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: Optional[torch.device | str] = None,
) -> float:
    """Compute unbiased MMD^2 with torch tensors."""
    dev = _resolve_torch_device(device)
    X = X.to(dev, dtype=torch.float32)
    Y = Y.to(dev, dtype=torch.float32)

    n = int(X.shape[0])
    m = int(Y.shape[0])
    if n < 2 or m < 2:
        return 0.0

    with torch.no_grad():
        Kxx = _rbf_kernel_torch(X, X)
        Kyy = _rbf_kernel_torch(Y, Y)
        Kxy = _rbf_kernel_torch(X, Y)

        Kxx.fill_diagonal_(0.0)
        Kyy.fill_diagonal_(0.0)

        term_xx = Kxx.sum() / float(n * (n - 1))
        term_yy = Kyy.sum() / float(m * (m - 1))
        term_xy = 2.0 * Kxy.sum() / float(n * m)
        out = term_xx + term_yy - term_xy

    return float(out.item())


# -----------------------
# Group-segment player core
# -----------------------
def compute_global_baseline_mean(X_all: np.ndarray) -> np.ndarray:
    """Compute the feature-wise mean replacement baseline."""
    if X_all.ndim == 3:
        flat = X_all.reshape(-1, X_all.shape[-1])
    else:
        flat = X_all
    return flat.mean(axis=0).astype(np.float32)


def apply_coalition(
    x_seq: np.ndarray,
    players: List[Dict],
    z: np.ndarray,
    baseline_mean: np.ndarray,
) -> np.ndarray:
    """Apply a coalition mask by replacing inactive player blocks."""
    x_seq = np.asarray(x_seq, dtype=np.float32)
    baseline_mean = np.asarray(baseline_mean, dtype=np.float32).reshape(-1)

    x_new = x_seq.copy()
    M = len(players)
    for m in range(M):
        if z[m] == 0:
            p = players[m]
            t0, t1 = p["time_range"]
            var_idx = p["var_indices"]
            x_new[t0:t1, var_idx] = baseline_mean[var_idx]
    return x_new


def _broadcast_baseline_to_seq(baseline_mean: np.ndarray, T: int) -> np.ndarray:
    """Broadcast a feature-wise baseline to a writable sequence."""
    baseline_mean = np.asarray(baseline_mean, dtype=np.float32).reshape(-1)
    return np.broadcast_to(baseline_mean[None, :], (T, baseline_mean.shape[0])).copy()


def shapley_for_one_sample(
    x_seq: np.ndarray,
    players: List[Dict],
    baseline_mean: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_permutations: int = 500,
    batch_size: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Estimate player attributions with batched permutation Shapley."""
    if rng is None:
        rng = np.random.default_rng()

    x_seq = np.asarray(x_seq, dtype=np.float32)
    baseline_mean = np.asarray(baseline_mean, dtype=np.float32).reshape(-1)

    T, D = x_seq.shape
    M = len(players)
    phi = np.zeros(M, dtype=np.float32)

    if M == 0:
        return phi

    # Preprocess player ranges for fast masking.
    t0s = np.empty(M, dtype=np.int32)
    t1s = np.empty(M, dtype=np.int32)
    var_lists: List[np.ndarray] = []

    for m, p in enumerate(players):
        t0, t1 = p["time_range"]
        t0s[m] = int(t0)
        t1s[m] = int(t1)
        var_lists.append(np.asarray(p["var_indices"], dtype=np.int64))

    # The empty coalition is the full baseline sequence.
    x_base = _broadcast_baseline_to_seq(baseline_mean, T)  # (T,D)
    f0 = float(predict_fn(x_base[None, ...])[0])

    bs = int(max(1, batch_size))
    # Pre-allocate the batch buffer to reduce allocation overhead.
    batch_buf = np.empty((bs, T, D), dtype=np.float32)
    idx_buf = np.empty((bs,), dtype=np.int32)

    for _ in range(int(num_permutations)):
        perm = rng.permutation(M)

        x_cur = x_base.copy()
        f_prev = f0

        fill = 0  # number of buffered samples

        def flush_buffer(curr_f_prev: float) -> float:
            nonlocal fill
            if fill <= 0:
                return curr_f_prev

            f_batch = predict_fn(batch_buf[:fill]).reshape(-1)
            for j in range(fill):
                midx = int(idx_buf[j])
                f_z = float(f_batch[j])
                phi[midx] += (f_z - curr_f_prev)
                curr_f_prev = f_z

            fill = 0
            return curr_f_prev

        for idx in perm:
            t0 = t0s[idx]
            t1 = t1s[idx]
            v = var_lists[idx]

            # Activate this player block in place.
            x_cur[t0:t1, v] = x_seq[t0:t1, v]

            # Store the current coalition for batched evaluation.
            batch_buf[fill] = x_cur
            idx_buf[fill] = int(idx)
            fill += 1

            if fill >= bs:
                f_prev = flush_buffer(f_prev)

        # Flush remaining coalitions.
        if fill > 0:
            f_prev = flush_buffer(f_prev)

    phi /= float(max(1, int(num_permutations)))
    return phi.astype(np.float32)


def shap_to_matrix(
    phi: np.ndarray,
    segments: List[Tuple[int, int]],
    groups: List[List[int]],
) -> np.ndarray:
    """Reshape group-major player attributions to a group-segment matrix."""
    num_groups = len(groups)
    num_segments = len(segments)
    assert len(phi) == num_groups * num_segments

    mat = np.zeros((num_groups, num_segments), dtype=np.float32)

    idx = 0
    for g_id in range(num_groups):
        for s_id in range(num_segments):
            mat[g_id, s_id] = float(phi[idx])
            idx += 1
    return mat
