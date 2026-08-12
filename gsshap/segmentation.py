# gsshap/segmentation.py
from typing import List, Optional, Tuple
import numpy as np
import torch
from .shapley import mmd2_unbiased, mmd2_unbiased_torch


def _calibrated_mmd_threshold(
    x_seq: np.ndarray,
    start: int,
    tau: int,
    end: int,
    alpha: float,
    num_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Estimate a split-specific null threshold by permuting time indices."""
    n_left = tau - start
    block = np.asarray(x_seq[start:end], dtype=np.float32)
    if n_left < 2 or end - tau < 2 or num_permutations <= 0:
        return 0.0

    vals = np.empty(int(num_permutations), dtype=np.float32)
    for i in range(int(num_permutations)):
        perm = rng.permutation(block.shape[0])
        left = block[perm[:n_left]]
        right = block[perm[n_left:]]
        vals[i] = float(mmd2_unbiased(left, right))

    q = float(np.clip(1.0 - float(alpha), 0.0, 1.0))
    return float(np.quantile(vals, q))


def segment_by_mmd(
    x_seq: np.ndarray,
    min_seg_len: int = 16,
    max_segments: int = 4,
    mmd_threshold: Optional[float] = None,
    threshold_alpha: float = 0.05,
    threshold_permutations: int = 50,
    candidate_stride: int = 1,
    use_torch: bool = False,
    device: str | torch.device | None = None,
    seed: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Recursive MMD segmentation with optional permutation-calibrated thresholds."""
    x_seq = np.asarray(x_seq, dtype=np.float32)
    T = x_seq.shape[0]
    segments: List[Tuple[int, int]] = [(0, T)]
    rng = np.random.default_rng(seed)
    torch_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_seq_torch = None
    if use_torch:
        x_seq_torch = torch.as_tensor(x_seq, dtype=torch.float32, device=torch_device)

    def _compute_mmd(start: int, tau: int, end: int) -> float:
        if use_torch and x_seq_torch is not None:
            left = x_seq_torch[start:tau]
            right = x_seq_torch[tau:end]
            return mmd2_unbiased_torch(left, right, device=torch_device)

        left = x_seq[start:tau]
        right = x_seq[tau:end]
        return mmd2_unbiased(left, right)

    def split_segment(start: int, end: int):
        nonlocal segments
        length = end - start
        if length < 2 * min_seg_len:
            return
        if len(segments) >= max_segments:
            return

        tau_start = start + min_seg_len
        tau_stop = end - min_seg_len
        if tau_start >= tau_stop:
            return

        candidate_stride_local = max(1, int(candidate_stride))
        coarse_taus = list(range(tau_start, tau_stop, candidate_stride_local))
        if not coarse_taus or coarse_taus[-1] != tau_stop - 1:
            coarse_taus.append(tau_stop - 1)

        best_mmd = 0.0
        best_tau = None
        for tau in coarse_taus:
            mmd_val = _compute_mmd(start, tau, end)
            if mmd_val > best_mmd:
                best_mmd = mmd_val
                best_tau = tau

        if best_tau is not None and candidate_stride_local > 1:
            refine_start = max(tau_start, best_tau - candidate_stride_local + 1)
            refine_stop = min(tau_stop, best_tau + candidate_stride_local)
            for tau in range(refine_start, refine_stop):
                mmd_val = _compute_mmd(start, tau, end)
                if mmd_val > best_mmd:
                    best_mmd = mmd_val
                    best_tau = tau

        if best_tau is None:
            return

        threshold = float(mmd_threshold) if mmd_threshold is not None else _calibrated_mmd_threshold(
            x_seq=x_seq,
            start=start,
            tau=best_tau,
            end=end,
            alpha=threshold_alpha,
            num_permutations=threshold_permutations,
            rng=rng,
        )

        if best_mmd > threshold:
            segments.remove((start, end))
            segments.append((start, best_tau))
            segments.append((best_tau, end))
            split_segment(start, best_tau)
            split_segment(best_tau, end)

    split_segment(0, T)
    segments = sorted(segments, key=lambda x: x[0])
    return segments
