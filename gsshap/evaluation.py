from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsshap.datasets import (
    load_ettm1_dataloaders,
    load_har_dataloaders,
    load_ptbxl_dataloaders,
    load_sp500_dataloaders,
)
from gsshap.explainers import build_group_segment_players, segment_groups_by_mmd
from gsshap.deletion import delete_cells_by_fraction, player_phi_to_cell_scores
from gsshap.hsic import cluster_features_hsic
from gsshap.models import BiLSTMSeqModel
from gsshap.shapley import compute_global_baseline_mean, shapley_for_one_sample

FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
DEFAULT_METHODS = ["joint"]
METHOD_LABELS = {"joint": "GS-SHAP"}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    task: str
    model_file: str
    hsic_max_samples: int
    min_seg_len: int
    max_segments: int
    mmd_threshold: Optional[float] = None
    mmd_alpha: float = 0.05
    mmd_permutations: int = 50
    candidate_stride: int = 1
    dropout: float = 0.2
    hidden_dim: int = 64
    num_layers: int = 2
    horizon_index: Optional[int] = None


DATASETS: Dict[str, DatasetSpec] = {
    "har": DatasetSpec("har", "HAR", "clf", "har_bilstm.pt", 3000, 10, 6),
    "ettm1": DatasetSpec("ettm1", "ETTm1", "reg", "ett_bilstm.pt", 3000, 13, 6),
    "ptbxl": DatasetSpec("ptbxl", "PTB-XL", "clf", "ptbxl_bilstm.pt", 3000, 100, 4),
    "sp500": DatasetSpec("sp500", "S&P500", "reg", "sp500_bilstm_h0.pt", 3000, 4, 4, horizon_index=0),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_bundle(spec: DatasetSpec, data_dir: str):
    if spec.key == "har":
        return load_har_dataloaders(data_dir, batch_size=128, val_ratio=0.1)
    if spec.key == "ettm1":
        return load_ettm1_dataloaders(data_dir, batch_size=128, val_ratio=0.1)
    if spec.key == "ptbxl":
        return load_ptbxl_dataloaders(data_dir, batch_size=128, val_ratio=0.1)
    if spec.key == "sp500":
        return load_sp500_dataloaders(data_dir, horizon_index=spec.horizon_index or 0, batch_size=128, val_ratio=0.1)
    raise KeyError(spec.key)


def build_model(spec: DatasetSpec, input_dim: int, output_dim: int, model_dir: str, device: torch.device) -> torch.nn.Module:
    model = BiLSTMSeqModel(
        input_dim=input_dim,
        hidden_dim=spec.hidden_dim,
        num_layers=spec.num_layers,
        output_dim=output_dim,
        dropout=spec.dropout,
        task=spec.task,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, spec.model_file), map_location=device))
    model.eval()
    return model


def make_score_functions(model: torch.nn.Module, spec: DatasetSpec, y_value, device: torch.device):
    """Return deletion-loss and explainer-output functions.

    The explainer score is aligned with the manuscript notation f(X):
      - classification: the target-class model logit,
      - regression: the scalar model prediction.

    The deletion metric is evaluated separately as prediction loss, matching
    Section 4.2 of the manuscript.
    """
    model.eval()
    if spec.task == "clf":
        target = int(y_value)

        def deletion_score(x_batch_np: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.asarray(x_batch_np, dtype=np.float32)).to(device)
            with torch.no_grad():
                return (-torch.log_softmax(model(x), dim=-1)[:, target]).detach().cpu().numpy()

        def explainer_score(x_batch_np: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.asarray(x_batch_np, dtype=np.float32)).to(device)
            with torch.no_grad():
                return model(x)[:, target].detach().cpu().numpy()

        return deletion_score, explainer_score, target

    y_true = float(y_value)

    def raw_prediction(x_batch_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(x_batch_np, dtype=np.float32)).to(device)
        with torch.no_grad():
            return model(x).view(x.shape[0], -1)[:, 0].detach().cpu().numpy()

    def deletion_score(x_batch_np: np.ndarray) -> np.ndarray:
        pred = raw_prediction(x_batch_np)
        return (pred - y_true) ** 2

    def explainer_score(x_batch_np: np.ndarray) -> np.ndarray:
        return raw_prediction(x_batch_np)

    return deletion_score, explainer_score, None


def shared_min_seg_len(spec: DatasetSpec, T: int) -> int:
    return int(spec.min_seg_len)


def build_feature_groups(full_train, spec: DatasetSpec, seed: int) -> List[List[int]]:
    X_flat = full_train.X.reshape(-1, full_train.X.shape[-1])
    return cluster_features_hsic(
        X_flat,
        max_samples=spec.hsic_max_samples,
        seed=seed,
    )

def compute_background_mean_from_sample(
    X_train: np.ndarray,
    *,
    max_samples: int = 3000,
    seed: int = 42,
) -> np.ndarray:
    """Feature-wise masking baseline from a fixed sampled background set.

    This matches Appendix E2: the background set is sampled from the
    training split with a fixed seed, and masking uses its feature-wise mean.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    n = int(X_train.shape[0])
    rng = np.random.RandomState(seed)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_bg = X_train[idx]
    else:
        X_bg = X_train
    return compute_global_baseline_mean(X_bg)


def empty_score_buffers(methods: Sequence[str], fractions: Sequence[float], num_samples: int):
    return {m: np.full((len(fractions), num_samples), np.nan, dtype=np.float32) for m in methods}


def store_scores(buffer: np.ndarray, fractions: Sequence[float], scores: Dict[float, float], sample_pos: int) -> None:
    for i, frac in enumerate(fractions):
        buffer[i, sample_pos] = float(scores[float(frac)])


def evaluate_sample(
    *,
    model: torch.nn.Module,
    spec: DatasetSpec,
    x_seq: np.ndarray,
    y_value,
    groups: List[List[int]],
    baseline_mean: np.ndarray,
    methods: Sequence[str],
    fractions: Sequence[float],
    budget_calls: int,
    min_perms: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[float, float]]:
    x_seq = np.asarray(x_seq, dtype=np.float32)
    T, D = x_seq.shape
    deletion_fn, score_fn, target_class = make_score_functions(model, spec, y_value, device)
    min_seg_len = shared_min_seg_len(spec, T)

    out: Dict[str, Dict[float, float]] = {}

    if "joint" in methods:
        segments_by_group = segment_groups_by_mmd(
            x_seq_np=x_seq,
            feature_groups=groups,
            min_seg_len=min_seg_len,
            max_segments=spec.max_segments,
            mmd_threshold=spec.mmd_threshold,
            threshold_alpha=spec.mmd_alpha,
            threshold_permutations=spec.mmd_permutations,
            candidate_stride=spec.candidate_stride,
            seed=seed,
        )
        players = build_group_segment_players(groups, segments_by_group)
        joint_perms = max(min_perms, budget_calls // max(1, len(players)))
        phi = shapley_for_one_sample(
            x_seq=x_seq,
            players=players,
            baseline_mean=baseline_mean,
            predict_fn=score_fn,
            num_permutations=joint_perms,
            rng=np.random.default_rng(seed),
        )
        cell = player_phi_to_cell_scores(phi, players, T, D)
        out["joint"] = delete_cells_by_fraction(
            x_seq, cell, list(fractions), deletion_fn, baseline_mean, use_abs=False
        )

    return out
