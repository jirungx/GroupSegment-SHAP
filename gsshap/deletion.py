# gsshap/deletion.py
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Optional, Union
import numpy as np


# ----------------------------------------------------------------------
# 0) Safe helpers
# ----------------------------------------------------------------------
def _to_scalar(pred_out: Union[float, np.ndarray, List[float]]) -> float:
    """Convert a scalar-like prediction output to a Python float."""
    if isinstance(pred_out, (float, int)):
        return float(pred_out)
    arr = np.asarray(pred_out)
    if arr.size == 0:
        raise ValueError("[deletion] predict_fn returned empty output.")
    return float(arr.reshape(-1)[0])


def _normalize_baseline_seq(
    baseline_seq_np: np.ndarray,
    T: int,
    D: int,
    name: str = "baseline_seq_np",
) -> np.ndarray:
    """Return a baseline sequence with shape (T, D)."""
    base = np.asarray(baseline_seq_np, dtype=np.float32)
    if base.ndim == 1 and base.shape[0] == D:
        base = np.tile(base[None, :], (T, 1))
    elif base.shape == (T, D):
        pass
    else:
        raise ValueError(
            f"{name} shape must be (D,) or (T,D). got {base.shape}, input (T,D)=({T},{D})"
        )
    return base.astype(np.float32)


def _predict_one(
    predict_fn: Callable[[np.ndarray], Union[float, np.ndarray, List[float]]],
    x_seq: np.ndarray,  # (T,D)
) -> float:
    """Evaluate one sequence with a batch-oriented prediction function."""
    return _to_scalar(predict_fn(x_seq[None, ...]))


def _trapz_auc(x: List[float], y: List[float]) -> float:
    auc = 0.0
    for i in range(1, len(x)):
        auc += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return float(auc)


# ----------------------------------------------------------------------
# Cell-level deletion
# ----------------------------------------------------------------------
def delete_cells_by_fraction(
    x_seq: np.ndarray,            # (T,D)
    phi_cell: np.ndarray,         # (T,D) importance map (signed or abs)
    fractions: List[float],
    predict_fn: Callable[[np.ndarray], Union[float, np.ndarray, List[float]]],
    baseline_seq_np: Optional[np.ndarray] = None,  # (D,) or (T,D); if None -> use zeros
    mode: str = "target_frac",    # target_frac | actual_frac
    use_abs: bool = True,
) -> Dict[float, float]:
    """Mask the highest-ranked cells and return scores at target fractions.

    The default mode records exactly the requested deletion fractions. The
    `actual_frac` mode records every newly reached masked-cell fraction.
    """
    x_seq = np.asarray(x_seq, dtype=np.float32)
    phi_cell = np.asarray(phi_cell, dtype=np.float32)

    T, D = x_seq.shape
    total_cells = T * D

    if phi_cell.shape != (T, D):
        raise ValueError(f"[delete_cells_by_fraction] phi_cell shape {phi_cell.shape} != {(T,D)}")

    # Resolve the masking baseline.
    if baseline_seq_np is None:
        baseline_seq = np.zeros((T, D), dtype=np.float32)
    else:
        baseline_seq = _normalize_baseline_seq(baseline_seq_np, T, D, name="baseline_seq_np")

    # Score before masking.
    orig_score = _predict_one(predict_fn, x_seq)

    # Rank cells by attribution magnitude or signed score.
    flat_importance = phi_cell.reshape(-1)
    if use_abs:
        flat_importance = np.abs(flat_importance)
    sorted_idx = np.argsort(-flat_importance)  # descending

    # Normalize the requested deletion grid.
    target_fracs = sorted(set(float(f) for f in fractions))
    if 0.0 not in target_fracs:
        target_fracs = [0.0] + target_fracs

    scores: Dict[float, float] = {}
    scores[0.0] = orig_score

    # Prepare flattened views for masking.
    masked = x_seq.copy().reshape(-1)
    base_flat = baseline_seq.reshape(-1)

    if mode == "target_frac":
        for frac in target_fracs:
            if frac <= 0.0:
                continue
            k = int(round(total_cells * frac))
            k = max(0, min(total_cells, k))
            if k == 0:
                scores[frac] = orig_score
                continue
            idx = sorted_idx[:k]
            tmp = masked.copy()
            tmp[idx] = base_flat[idx]
            score = _to_scalar(predict_fn(tmp.reshape(1, T, D)))
            scores[frac] = score
        return scores

    if mode == "actual_frac":
        # Incrementally mask cells and record each reached fraction.
        cell_mask = np.zeros(total_cells, dtype=bool)
        # Record the unmasked score.
        scores = {0.0: orig_score}

        for k in range(1, total_cells + 1):
            idx = int(sorted_idx[k - 1])
            if cell_mask[idx]:
                continue
            cell_mask[idx] = True
            masked[idx] = base_flat[idx]

            # Record the current masked state.
            frac_now = float(cell_mask.sum() / total_cells)
            if frac_now in scores:
                continue
            # Dense recording is useful for later interpolation.
            scores[frac_now] = _to_scalar(predict_fn(masked.reshape(1, T, D)))

        return scores

    raise ValueError('[delete_cells_by_fraction] mode must be "target_frac" or "actual_frac".')


# ----------------------------------------------------------------------
# Player-level deletion
# ----------------------------------------------------------------------
def delete_by_player_blocks(
    x_seq: np.ndarray,                    # (T,D)
    player_phi: np.ndarray,               # (P,)
    player_blocks: Sequence[np.ndarray],  # len=P, each: 1D flattened indices
    fractions: List[float],
    predict_fn: Callable[[np.ndarray], Union[float, np.ndarray, List[float]]],
    baseline_seq_np: Optional[np.ndarray] = None,  # (D,) or (T,D); if None -> zeros
    mode: str = "target_frac",            # target_frac | actual_frac
) -> Dict[float, float]:
    """Mask players in attribution order and return scores at deletion fractions."""
    x_seq = np.asarray(x_seq, dtype=np.float32)
    player_phi = np.asarray(player_phi, dtype=np.float32)

    T, D = x_seq.shape
    total_cells = T * D

    P = int(player_phi.shape[0])
    if len(player_blocks) != P:
        raise ValueError(
            f"[delete_by_player_blocks] len(player_blocks)={len(player_blocks)} != len(player_phi)={P}"
        )

    # Resolve the masking baseline.
    if baseline_seq_np is None:
        baseline_seq = np.zeros((T, D), dtype=np.float32)
    else:
        baseline_seq = _normalize_baseline_seq(baseline_seq_np, T, D, name="baseline_seq_np")

    base_flat = baseline_seq.reshape(-1)

    # Sort players by attribution magnitude.
    abs_phi = np.abs(player_phi).astype(np.float32)
    sorted_players = np.argsort(-abs_phi)  # descending

    # Normalize the requested deletion grid.
    target_fracs = sorted(set(float(f) for f in fractions))
    if 0.0 not in target_fracs:
        target_fracs = [0.0] + target_fracs

    # Score before masking.
    orig_score = _predict_one(predict_fn, x_seq)

    masked = x_seq.copy().reshape(-1)
    cell_mask = np.zeros(total_cells, dtype=bool)

    if mode == "actual_frac":
        scores: Dict[float, float] = {0.0: orig_score}

        for p in sorted_players:
            blk = np.asarray(player_blocks[p], dtype=int).reshape(-1)
            if blk.size == 0:
                continue
            # Keep only cells not masked by previous players.
            blk = blk[(blk >= 0) & (blk < total_cells)]
            new = blk[~cell_mask[blk]]
            if new.size == 0:
                continue
            cell_mask[new] = True
            masked[new] = base_flat[new]

            frac_now = float(cell_mask.sum() / total_cells)
            # Record the current masked state.
            scores[frac_now] = _to_scalar(predict_fn(masked.reshape(1, T, D)))

            if frac_now >= 1.0 - 1e-12:
                break

        return scores

    if mode == "target_frac":
        scores: Dict[float, float] = {0.0: orig_score}
        next_idx = 1

        for p in sorted_players:
            blk = np.asarray(player_blocks[p], dtype=int).reshape(-1)
            if blk.size == 0:
                continue
            blk = blk[(blk >= 0) & (blk < total_cells)]
            new = blk[~cell_mask[blk]]
            if new.size == 0:
                continue

            cell_mask[new] = True
            masked[new] = base_flat[new]
            frac_now = float(cell_mask.sum() / total_cells)

            # Evaluate once for each newly crossed target fraction.
            while next_idx < len(target_fracs) and frac_now >= target_fracs[next_idx] - 1e-12:
                fr = target_fracs[next_idx]
                score = _to_scalar(predict_fn(masked.reshape(1, T, D)))
                scores[fr] = score
                next_idx += 1
                if next_idx >= len(target_fracs):
                    break

            if next_idx >= len(target_fracs):
                break

        # Fill any unreached targets with the final masked state.
        if next_idx < len(target_fracs):
            final_score = _to_scalar(predict_fn(masked.reshape(1, T, D)))
            for i in range(next_idx, len(target_fracs)):
                scores[target_fracs[i]] = final_score

        return scores

    raise ValueError('[delete_by_player_blocks] mode must be "target_frac" or "actual_frac".')


# ----------------------------------------------------------------------
# Deletion AUC
# ----------------------------------------------------------------------
def auc_curve(
    fractions: List[float],
    scores: Dict[float, float],
    kind: str = "raw",  # raw | drop | remaining
    orig_score: Optional[float] = None,
) -> float:
    """Compute AUC for a deletion curve.

    `raw` integrates the score itself, `drop` integrates score degradation,
    and `remaining` integrates the score normalized by the unmasked value.
    """
    fracs_sorted = sorted(set(float(f) for f in fractions))
    if 0.0 not in fracs_sorted:
        fracs_sorted = [0.0] + fracs_sorted

    if orig_score is None:
        if 0.0 not in scores:
            raise ValueError("[auc_curve] orig_score is None and scores has no key 0.0.")
        orig_score = float(scores[0.0])

    y = []
    for f in fracs_sorted:
        y.append(float(scores.get(f, scores.get(0.0, orig_score))))

    if kind == "raw":
        return _trapz_auc(fracs_sorted, y)

    if kind == "drop":
        yd = [float(orig_score) - v for v in y]
        return _trapz_auc(fracs_sorted, yd)

    if kind == "remaining":
        if orig_score == 0:
            raise ValueError("[auc_curve] remaining AUC undefined when orig_score == 0.")
        yr = [v / float(orig_score) for v in y]
        return _trapz_auc(fracs_sorted, yr)

    raise ValueError('[auc_curve] kind must be "raw", "drop", or "remaining".')


# ----------------------------------------------------------------------
# Block-to-cell projection
# Convert block attributions to a common (T, D) map for deletion.
# ----------------------------------------------------------------------
def _block_phi_to_cell_scores(
    blocks: List[Tuple[int, int, List[int]]],
    phi_block: np.ndarray,
    T: int,
    D: int,
) -> np.ndarray:
    """
    Assign each cell the average phi of all blocks containing it.
    For non-overlapping blocks (the typical case for segment-based methods),
    each cell simply receives its block's phi value unchanged.
    """
    scores = np.zeros((T, D), dtype=np.float64)
    counts = np.zeros((T, D), dtype=np.float64)
    phi_block = np.asarray(phi_block, dtype=np.float64).reshape(-1)

    for i, (s, e, feats) in enumerate(blocks):
        val = float(phi_block[i])
        denom = max(1, (e - s) * max(1, len(feats)))
        val = val / float(denom)
        for d in feats:
            scores[s:e, d] += val
            counts[s:e, d] += 1.0

    denom = np.maximum(1.0, counts)
    return (scores / denom).astype(np.float32)


def player_phi_to_cell_scores(
    player_phi: np.ndarray,
    players: Sequence[Dict],
    T: int,
    D: int,
) -> np.ndarray:
    """Project arbitrary group-segment player scores to a dense cell map."""
    blocks: List[Tuple[int, int, List[int]]] = []
    phi_vals: List[float] = []
    phi = np.asarray(player_phi, dtype=np.float64).reshape(-1)
    if len(players) != phi.shape[0]:
        raise ValueError(f"len(players)={len(players)} != len(player_phi)={phi.shape[0]}")

    for i, player in enumerate(players):
        start, end = player["time_range"]
        feats = [int(d) for d in player["var_indices"]]
        blocks.append((int(start), int(end), feats))
        phi_vals.append(float(phi[i]))

    return _block_phi_to_cell_scores(blocks, np.asarray(phi_vals), T, D)
