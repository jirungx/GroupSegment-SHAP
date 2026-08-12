#!/usr/bin/env python
# coding: utf-8
"""GS-SHAP explainer implementation.

This module contains only the proposed GroupSegment-SHAP explainer used by
this code release. Baseline explainer implementations are intentionally not
included in this archive.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .segmentation import segment_by_mmd
from .shapley import shapley_for_one_sample


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_regression_model(
    model: torch.nn.Module,
    x_seq_np: np.ndarray,
    device: Optional[torch.device] = None,
) -> bool:
    """Infer whether a model returns a scalar regression output."""
    device = _get_device(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(x_seq_np[None, ...]).float().to(device)
        out = model(x)

    if out.dim() == 1:
        return True
    if out.dim() == 2 and out.shape[-1] == 1:
        return True
    return False


def _build_np_batch_pred_fn(
    model: torch.nn.Module,
    is_regression: bool,
    target_class: Optional[int],
    device: Optional[torch.device] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a batch scoring function aligned with the paper notation f(X).

    - regression: scalar model prediction
    - classification: target-class model logit
    """
    device = _get_device(device)

    def pred_fn(x_batch_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x_batch_np, dtype=np.float32)).float().to(device)
        with torch.no_grad():
            out = model(x)
        if is_regression:
            out = out.view(out.shape[0], -1)
            return out[:, 0].detach().cpu().numpy()
        if target_class is None:
            raise ValueError("Classification requires target_class.")
        return out[:, target_class].detach().cpu().numpy()

    return pred_fn


def _resolve_score_fn(
    model: torch.nn.Module,
    x_seq_np: np.ndarray,
    target_class: Optional[int],
    device: Optional[torch.device],
    score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve the model-output function attributed by GS-SHAP."""
    if score_fn is not None:
        return score_fn

    is_reg = _is_regression_model(model, x_seq_np, device=device)
    return _build_np_batch_pred_fn(model, is_reg, target_class, device)


def build_group_segment_players(
    feature_groups: List[List[int]],
    segments_by_group: List[List[Tuple[int, int]]],
) -> List[Dict]:
    """Build players p_{k,j}=G_k x S_j^(k) in group-major order."""
    if len(feature_groups) != len(segments_by_group):
        raise ValueError("feature_groups and segments_by_group must have the same length.")

    players: List[Dict] = []
    for group_id, (group, segments) in enumerate(zip(feature_groups, segments_by_group)):
        for segment_id, (start, end) in enumerate(segments):
            players.append(
                {
                    "group_id": int(group_id),
                    "segment_id": int(segment_id),
                    "var_indices": [int(d) for d in group],
                    "time_range": (int(start), int(end)),
                }
            )
    return players


def segment_groups_by_mmd(
    x_seq_np: np.ndarray,
    feature_groups: List[List[int]],
    min_seg_len: int,
    max_segments: int,
    mmd_threshold: Optional[float] = None,
    threshold_alpha: float = 0.05,
    threshold_permutations: int = 50,
    candidate_stride: int = 1,
    seed: Optional[int] = None,
) -> List[List[Tuple[int, int]]]:
    """Run MMD segmentation separately for each HSIC feature group."""
    rng = np.random.default_rng(seed)
    segments_by_group: List[List[Tuple[int, int]]] = []
    for group in feature_groups:
        group_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        x_group = np.asarray(x_seq_np[:, list(group)], dtype=np.float32)
        segments_by_group.append(
            segment_by_mmd(
                x_group,
                min_seg_len=min_seg_len,
                max_segments=max_segments,
                mmd_threshold=mmd_threshold,
                threshold_alpha=threshold_alpha,
                threshold_permutations=threshold_permutations,
                candidate_stride=candidate_stride,
                seed=group_seed,
            )
        )
    return segments_by_group


def groupsegmentshap_importance(
    model: torch.nn.Module,
    x_seq_np: np.ndarray,
    feature_groups: List[List[int]],
    baseline_mean: np.ndarray,
    target_class: Optional[int] = None,
    score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    min_seg_len: int = 16,
    max_segments: int = 4,
    mmd_threshold: Optional[float] = None,
    threshold_alpha: float = 0.05,
    threshold_permutations: int = 50,
    candidate_stride: int = 1,
    num_permutations: int = 300,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[Dict], List[List[Tuple[int, int]]]]:
    """Compute GS-SHAP with group-specific MMD temporal segments."""
    device = _get_device(device)
    model.eval()
    pred_fn = _resolve_score_fn(
        model=model,
        x_seq_np=x_seq_np,
        target_class=target_class,
        device=device,
        score_fn=score_fn,
    )
    segments_by_group = segment_groups_by_mmd(
        x_seq_np=x_seq_np,
        feature_groups=feature_groups,
        min_seg_len=min_seg_len,
        max_segments=max_segments,
        mmd_threshold=mmd_threshold,
        threshold_alpha=threshold_alpha,
        threshold_permutations=threshold_permutations,
        candidate_stride=candidate_stride,
        seed=seed,
    )
    players = build_group_segment_players(feature_groups, segments_by_group)
    phi = shapley_for_one_sample(
        x_seq=x_seq_np,
        players=players,
        baseline_mean=baseline_mean,
        predict_fn=pred_fn,
        num_permutations=num_permutations,
        rng=np.random.default_rng(seed),
    )
    return phi, players, segments_by_group
