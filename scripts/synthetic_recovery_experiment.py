#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gsshap.training import train_sequence_classifier
from gsshap.explainers import build_group_segment_players, segment_groups_by_mmd
from gsshap.deletion import player_phi_to_cell_scores
from gsshap.hsic import cluster_features_hsic
from gsshap.models import build_sequence_model
from gsshap.shapley import shapley_for_one_sample


METHODS = ["joint"]
DEFAULT_METHODS = ["joint"]
METHOD_LABELS = {"joint": "GS-SHAP"}
SETTING_LABELS = {
    "A": "Setting A: Single-Player Signal",
    "B": "Setting B: Two-Player Additive Signal",
    "C": "Setting C: Distractor Group Signal",
}


@dataclass(frozen=True)
class SyntheticConfig:
    name: str
    num_train: int = 5000
    num_val: int = 1000
    num_test: int = 1000
    seq_len: int = 100
    num_features: int = 12
    num_groups: int = 4
    gt_segments_per_sample: int = 5
    explainer_segments_per_sample: int = 10
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.3
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 128
    budget_calls: int = 1500
    hsic_max_samples: int = 3000
    eval_samples: int = 100
    num_seeds: int = 3


@dataclass
class SyntheticSplit:
    X: np.ndarray
    y: np.ndarray
    gt_players: List[List[Tuple[int, int]]]
    gt_masks: np.ndarray


@dataclass
class SyntheticBundle:
    config: SyntheticConfig
    true_groups: List[List[int]]
    true_segments: List[Tuple[int, int]]
    train: SyntheticSplit
    val: SyntheticSplit
    test: SyntheticSplit


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic ground-truth recovery experiment for GS-SHAP."
    )
    parser.add_argument("--settings", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train", type=int, default=5000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--budget-calls", type=int, default=1500)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--hsic-max-samples", type=int, default=3000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--reuse-models", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny synthetic smoke test without writing full experiment outputs.")
    parser.add_argument("--append-existing", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "synthetic_ground_truth_recovery"),
    )
    return parser.parse_args()


def fixed_groups(num_features: int, num_groups: int) -> List[List[int]]:
    feats_per_group = num_features // num_groups
    groups: List[List[int]] = []
    for g in range(num_groups):
        start = g * feats_per_group
        groups.append(list(range(start, start + feats_per_group)))
    return groups


def fixed_segments(seq_len: int, num_segments: int) -> List[Tuple[int, int]]:
    base = seq_len // num_segments
    rem = seq_len % num_segments
    out: List[Tuple[int, int]] = []
    start = 0
    for s in range(num_segments):
        seg_len = base + (1 if s < rem else 0)
        end = start + seg_len
        out.append((start, end))
        start = end
    return out


def ar1_process(rng: np.random.RandomState, T: int, rho: float, sigma: float) -> np.ndarray:
    x = np.zeros((T,), dtype=np.float32)
    x[0] = rng.normal(scale=sigma)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(scale=sigma)
    return x


def pulse_profile(length: int) -> np.ndarray:
    grid = np.linspace(0.0, math.pi, num=length, dtype=np.float32)
    return (0.5 + np.sin(grid)).astype(np.float32)


def choose_players(
    setting: str,
    rng: np.random.RandomState,
    num_groups: int,
    num_segments: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if setting == "A":
        gt = [(int(rng.randint(num_groups)), int(rng.randint(num_segments)))]
        return gt, []

    if setting == "B":
        g1 = int(rng.randint(num_groups))
        g2_choices = [g for g in range(num_groups) if g != g1]
        g2 = int(rng.choice(g2_choices))
        s1 = int(rng.choice([0, 1]))
        s2 = int(rng.choice([3, 4]))
        return [(g1, s1), (g2, s2)], []

    gt_group = int(rng.randint(num_groups))
    gt_seg = int(rng.randint(num_segments))
    distractor_choices = [g for g in range(num_groups) if g != gt_group]
    distractor_group = int(rng.choice(distractor_choices))
    return [(gt_group, gt_seg)], [(distractor_group, gt_seg)]


def make_ground_truth_mask(
    players: Sequence[Tuple[int, int]],
    groups: Sequence[Sequence[int]],
    segments: Sequence[Tuple[int, int]],
    T: int,
    D: int,
) -> np.ndarray:
    mask = np.zeros((T, D), dtype=np.float32)
    for g_idx, s_idx in players:
        t0, t1 = segments[int(s_idx)]
        feats = groups[int(g_idx)]
        mask[t0:t1, feats] = 1.0
    return mask


def generate_split(
    setting: str,
    num_samples: int,
    groups: Sequence[Sequence[int]],
    segments: Sequence[Tuple[int, int]],
    seed: int,
) -> SyntheticSplit:
    rng = np.random.RandomState(seed)
    T = segments[-1][1]
    D = sum(len(g) for g in groups)

    X = np.zeros((num_samples, T, D), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.int64)
    gt_players: List[List[Tuple[int, int]]] = []
    gt_masks = np.zeros((num_samples, T, D), dtype=np.float32)

    for i in range(num_samples):
        label = int(rng.randint(2))
        class_sign = 1.0 if label == 1 else -1.0

        sample = np.zeros((T, D), dtype=np.float32)
        for g_idx, feats in enumerate(groups):
            latent = ar1_process(rng, T=T, rho=0.8, sigma=0.22)
            low_freq = 0.15 * np.sin(np.linspace(0.0, 2.0 * math.pi, T, dtype=np.float32) + 0.4 * g_idx)
            group_signal = latent + low_freq.astype(np.float32)
            for feat in feats:
                sample[:, feat] = group_signal + rng.normal(scale=0.10, size=T).astype(np.float32)

        active_players, distractors = choose_players(setting, rng, len(groups), len(segments))

        for rank, (g_idx, s_idx) in enumerate(active_players):
            t0, t1 = segments[s_idx]
            profile = pulse_profile(t1 - t0)
            amp = 3.0 if setting != "B" else (3.0 - 0.4 * rank)
            sample[t0:t1, groups[g_idx]] += class_sign * amp * profile[:, None]

        for g_idx, s_idx in distractors:
            t0, t1 = segments[s_idx]
            profile = pulse_profile(t1 - t0)
            nuisance_sign = -1.0 if rng.rand() < 0.5 else 1.0
            sample[t0:t1, groups[g_idx]] += nuisance_sign * 1.2 * profile[:, None]
            sample[t0:t1, groups[g_idx]] += rng.normal(scale=0.10, size=(t1 - t0, len(groups[g_idx]))).astype(np.float32)

        # Mild global nuisance pattern that should not dominate local explanations.
        sample += 0.03 * rng.normal(size=sample.shape).astype(np.float32)

        X[i] = sample
        y[i] = label
        gt_players.append(list(active_players))
        gt_masks[i] = make_ground_truth_mask(active_players, groups, segments, T, D)

    return SyntheticSplit(X=X, y=y, gt_players=gt_players, gt_masks=gt_masks)


def build_synthetic_bundle(setting: str, cfg: SyntheticConfig, seed: int) -> SyntheticBundle:
    true_groups = fixed_groups(cfg.num_features, cfg.num_groups)
    true_segments = fixed_segments(cfg.seq_len, cfg.gt_segments_per_sample)
    train = generate_split(setting, cfg.num_train, true_groups, true_segments, seed + 11)
    val = generate_split(setting, cfg.num_val, true_groups, true_segments, seed + 23)
    test = generate_split(setting, cfg.num_test, true_groups, true_segments, seed + 37)
    return SyntheticBundle(
        config=cfg,
        true_groups=true_groups,
        true_segments=true_segments,
        train=train,
        val=val,
        test=test,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def predict_positive_proba(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for X, _ in loader:
            logits = model(X.to(device))
            probs = torch.softmax(logits, dim=-1)[:, 1]
            outs.append(probs.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def compute_background_mean(X: np.ndarray, max_samples: int = 3000, seed: int = 42) -> np.ndarray:
    """Feature-wise mean over a fixed sampled training background set."""
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.RandomState(seed)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
    return X.reshape(-1, X.shape[-1]).mean(axis=0).astype(np.float32)


def make_clf_utility_fn(model: torch.nn.Module, device: torch.device, target_class: int):
    """Target-class model output used for Shapley attribution.

    This follows the manuscript's f(X) notation by attributing the raw
    target-class logit rather than a loss or log-probability transform.
    """
    model.eval()

    def _fn(x_batch_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_batch_np).float().to(device)
        with torch.no_grad():
            logits = model(x)
            return logits[:, target_class].detach().cpu().numpy()

    return _fn


def aggregate_cell_scores_to_players(
    phi_cell: np.ndarray,
    groups: Sequence[Sequence[int]],
    segments: Sequence[Tuple[int, int]],
) -> np.ndarray:
    scores = np.zeros((len(groups), len(segments)), dtype=np.float32)
    phi_abs = np.abs(np.asarray(phi_cell, dtype=np.float32))
    for g_idx, feats in enumerate(groups):
        for s_idx, (t0, t1) in enumerate(segments):
            block = phi_abs[t0:t1, feats]
            scores[g_idx, s_idx] = float(block.mean()) if block.size > 0 else 0.0
    return scores


def topk_cell_mask(phi_cell: np.ndarray, k: int) -> np.ndarray:
    phi_abs = np.abs(np.asarray(phi_cell, dtype=np.float32))
    flat = phi_abs.reshape(-1)
    mask = np.zeros_like(flat, dtype=np.float32)
    if k <= 0:
        return mask.reshape(phi_abs.shape)
    top_idx = np.argpartition(flat, -k)[-k:]
    mask[top_idx] = 1.0
    return mask.reshape(phi_abs.shape)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def player_metrics(
    player_scores: np.ndarray,
    gt_players: Sequence[Tuple[int, int]],
) -> Dict[str, float]:
    flat_scores = player_scores.reshape(-1)
    num_players = flat_scores.size
    gt_ids = {int(g) * player_scores.shape[1] + int(s) for g, s in gt_players}
    k = max(1, len(gt_ids))

    order = np.argsort(-flat_scores)
    top1 = int(order[0]) if num_players > 0 else -1
    top3 = set(int(idx) for idx in order[: min(3, num_players)])
    topk = set(int(idx) for idx in order[:k])

    group_scores = player_scores.sum(axis=1)
    seg_scores = player_scores.sum(axis=0)
    pred_group = int(np.argmax(group_scores))
    pred_seg = int(np.argmax(seg_scores))
    gt_groups = {int(g) for g, _ in gt_players}
    gt_segs = {int(s) for _, s in gt_players}

    union = len(topk | gt_ids)
    player_iou = 0.0 if union == 0 else len(topk & gt_ids) / float(union)

    return {
        "player_top1_hit": float(top1 in gt_ids),
        "player_top3_hit": float(len(top3 & gt_ids) > 0),
        "player_iou": float(player_iou),
        "group_top1_hit": float(pred_group in gt_groups),
        "segment_top1_hit": float(pred_seg in gt_segs),
    }


def evaluate_method_on_sample(
    method: str,
    model: torch.nn.Module,
    x_seq_np: np.ndarray,
    y_true: int,
    baseline_mean: np.ndarray,
    inferred_groups: Sequence[Sequence[int]],
    explainer_segments: Sequence[Tuple[int, int]],
    cfg: SyntheticConfig,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    if method != "joint":
        raise ValueError("This code release contains only the proposed GS-SHAP method.")

    utility_fn = make_clf_utility_fn(model, device, target_class=int(y_true))
    t0 = time.perf_counter()
    x_seq_np = np.asarray(x_seq_np, dtype=np.float32)
    budget_calls = max(50, int(cfg.budget_calls))

    groups = [list(g) for g in inferred_groups]
    min_seg_len = max(2, explainer_segments[0][1] - explainer_segments[0][0])
    segments_by_group = segment_groups_by_mmd(
        x_seq_np=x_seq_np,
        feature_groups=groups,
        min_seg_len=min_seg_len,
        max_segments=len(explainer_segments),
        mmd_threshold=None,
        threshold_alpha=0.05,
        threshold_permutations=30,
        candidate_stride=1,
        seed=seed,
    )
    players = build_group_segment_players(groups, segments_by_group)
    joint_permutations = max(1, budget_calls // max(1, len(players)))
    phi_joint = shapley_for_one_sample(
        x_seq=x_seq_np,
        players=players,
        baseline_mean=baseline_mean,
        predict_fn=utility_fn,
        num_permutations=joint_permutations,
        rng=np.random.default_rng(seed),
    )
    phi_cell = player_phi_to_cell_scores(
        phi_joint,
        players,
        x_seq_np.shape[0],
        x_seq_np.shape[1],
    )
    return np.asarray(phi_cell, dtype=np.float32), float(time.perf_counter() - t0)


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: str, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_csv_rows(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    numeric_keys = {
        "best_val_acc",
        "test_acc",
        "test_auroc",
        "cell_iou",
        "player_iou",
        "player_top1_hit",
        "player_top3_hit",
        "group_top1_hit",
        "segment_top1_hit",
        "avg_runtime_sec",
        "num_eval_samples",
        "num_runs",
    }
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in numeric_keys:
            if key in row and row[key] not in (None, ""):
                if key.startswith("num_"):
                    row[key] = int(float(row[key]))
                else:
                    row[key] = float(row[key])
    return rows


def read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_qualitative_example(
    out_path: str,
    setting: str,
    sample_idx: int,
    x_seq: np.ndarray,
    gt_mask: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
) -> None:
    panels = [("Ground Truth", gt_mask)] + [(METHOD_LABELS[m], np.abs(mat)) for m, mat in heatmaps.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.4), constrained_layout=True)
    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat.T, aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{SETTING_LABELS[setting]} | sample={sample_idx}")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_predictor(
    bundle: SyntheticBundle,
    cfg: SyntheticConfig,
    device: torch.device,
    out_dir: str,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    train_loader = make_loader(bundle.train.X, bundle.train.y, cfg.batch_size, shuffle=True)
    val_loader = make_loader(bundle.val.X, bundle.val.y, cfg.batch_size, shuffle=False)
    test_loader = make_loader(bundle.test.X, bundle.test.y, cfg.batch_size, shuffle=False)
    model_path = os.path.join(out_dir, f"model_{bundle.config.name}.pt")
    log_path = os.path.join(out_dir, f"train_{bundle.config.name}.csv")

    model, best_val_acc, test_acc = train_sequence_classifier(
        backbone="bilstm",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_dim=bundle.config.num_features,
        num_classes=2,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_epochs=cfg.epochs,
        lr=cfg.lr,
        device=device,
        model_path=model_path,
        log_path=log_path,
        dropout=cfg.dropout,
    )

    test_probs = predict_positive_proba(model, test_loader, device=device)
    test_auroc = binary_auroc(bundle.test.y, test_probs)
    return model, {
        "setting": bundle.config.name,
        "setting_label": SETTING_LABELS[bundle.config.name],
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_auroc": float(test_auroc),
    }


def load_trained_model(cfg: SyntheticConfig, device: torch.device, model_path: str) -> torch.nn.Module:
    model = build_sequence_model(
        backbone="bilstm",
        input_dim=cfg.num_features,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=2,
        dropout=cfg.dropout,
        task="clf",
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_recovery(
    bundle: SyntheticBundle,
    model: torch.nn.Module,
    cfg: SyntheticConfig,
    device: torch.device,
    out_dir: str,
    seed: int,
    inferred_groups: Sequence[Sequence[int]],
    explainer_segments: Sequence[Tuple[int, int]],
    run_tag: str,
    methods: Sequence[str],
) -> List[Dict[str, object]]:
    baseline_mean = compute_background_mean(bundle.train.X, max_samples=cfg.hsic_max_samples, seed=seed)
    rng = np.random.RandomState(seed + 101)
    num_eval = min(cfg.eval_samples, bundle.test.X.shape[0])
    eval_indices = np.sort(rng.choice(bundle.test.X.shape[0], size=num_eval, replace=False))

    rows: List[Dict[str, object]] = []
    qualitative_saved = False

    per_method: Dict[str, Dict[str, List[float]]] = {
        method: {
            "cell_iou": [],
            "player_iou": [],
            "player_top1_hit": [],
            "player_top3_hit": [],
            "group_top1_hit": [],
            "segment_top1_hit": [],
            "runtime_sec": [],
        }
        for method in methods
    }

    sample_iter = tqdm(
        eval_indices,
        desc=f"Recovery {bundle.config.name}/{run_tag}",
        unit="sample",
        leave=False,
    )
    for local_pos, sample_idx in enumerate(sample_iter):
        x_seq = bundle.test.X[int(sample_idx)]
        y_true = int(bundle.test.y[int(sample_idx)])
        gt_mask = bundle.test.gt_masks[int(sample_idx)]
        gt_players = bundle.test.gt_players[int(sample_idx)]

        heatmaps: Dict[str, np.ndarray] = {}

        for method in methods:
            phi_cell, elapsed = evaluate_method_on_sample(
                method=method,
                model=model,
                x_seq_np=x_seq,
                y_true=y_true,
                baseline_mean=baseline_mean,
                inferred_groups=inferred_groups,
                explainer_segments=explainer_segments,
                cfg=cfg,
                seed=seed + 1000 * (local_pos + 1),
                device=device,
            )
            player_scores = aggregate_cell_scores_to_players(
                phi_cell,
                bundle.true_groups,
                bundle.true_segments,
            )
            pred_cell_mask = topk_cell_mask(phi_cell, int(np.sum(gt_mask > 0)))
            sample_metrics = player_metrics(player_scores, gt_players)

            per_method[method]["cell_iou"].append(iou(pred_cell_mask, gt_mask))
            per_method[method]["player_iou"].append(sample_metrics["player_iou"])
            per_method[method]["player_top1_hit"].append(sample_metrics["player_top1_hit"])
            per_method[method]["player_top3_hit"].append(sample_metrics["player_top3_hit"])
            per_method[method]["group_top1_hit"].append(sample_metrics["group_top1_hit"])
            per_method[method]["segment_top1_hit"].append(sample_metrics["segment_top1_hit"])
            per_method[method]["runtime_sec"].append(elapsed)
            heatmaps[method] = phi_cell

        if not qualitative_saved:
            plot_qualitative_example(
                out_path=os.path.join(out_dir, f"qualitative_{bundle.config.name}_{run_tag}.png"),
                setting=bundle.config.name,
                sample_idx=int(sample_idx),
                x_seq=x_seq,
                gt_mask=gt_mask,
                heatmaps=heatmaps,
            )
            qualitative_saved = True

    for method in methods:
        rows.append(
            {
                "setting": bundle.config.name,
                "setting_label": SETTING_LABELS[bundle.config.name],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "cell_iou": float(np.mean(per_method[method]["cell_iou"])),
                "player_iou": float(np.mean(per_method[method]["player_iou"])),
                "player_top1_hit": float(np.mean(per_method[method]["player_top1_hit"])),
                "player_top3_hit": float(np.mean(per_method[method]["player_top3_hit"])),
                "group_top1_hit": float(np.mean(per_method[method]["group_top1_hit"])),
                "segment_top1_hit": float(np.mean(per_method[method]["segment_top1_hit"])),
                "avg_runtime_sec": float(np.mean(per_method[method]["runtime_sec"])),
                "num_eval_samples": int(num_eval),
                "seed": run_tag,
            }
        )

    return rows


def save_predictor_tables(rows: List[Dict[str, object]], out_dir: str) -> None:
    csv_path = os.path.join(out_dir, "predictor_performance.csv")
    fieldnames = ["setting", "setting_label", "seed", "best_val_acc", "test_acc", "test_auroc"]
    write_csv(csv_path, rows, fieldnames)

    md_rows = [
        [
            str(row["setting_label"]),
            f'{row["best_val_acc"]:.4f}',
            f'{row["test_acc"]:.4f}',
            f'{row["test_auroc"]:.4f}',
        ]
        for row in rows
    ]
    write_markdown(
        os.path.join(out_dir, "predictor_performance.md"),
        ["Setting", "Best Val Acc", "Test Acc", "Test AUROC"],
        md_rows,
    )


def save_recovery_tables(rows: List[Dict[str, object]], out_dir: str) -> None:
    fieldnames = [
        "setting",
        "setting_label",
        "seed",
        "method",
        "method_label",
        "cell_iou",
        "player_iou",
        "player_top1_hit",
        "player_top3_hit",
        "group_top1_hit",
        "segment_top1_hit",
        "avg_runtime_sec",
        "num_eval_samples",
    ]
    write_csv(os.path.join(out_dir, "ground_truth_recovery.csv"), rows, fieldnames)

    headers = [
        "Setting",
        "Method",
        "Cell IoU",
        "Player IoU",
        "Top-1 Hit",
        "Top-3 Hit",
        "Group Hit",
        "Segment Hit",
        "Avg Runtime (s)",
    ]
    md_rows = [
        [
            str(row["setting_label"]),
            str(row["method_label"]),
            f'{row["cell_iou"]:.4f}',
            f'{row["player_iou"]:.4f}',
            f'{row["player_top1_hit"]:.4f}',
            f'{row["player_top3_hit"]:.4f}',
            f'{row["group_top1_hit"]:.4f}',
            f'{row["segment_top1_hit"]:.4f}',
            f'{row["avg_runtime_sec"]:.4f}',
        ]
        for row in rows
    ]
    write_markdown(os.path.join(out_dir, "ground_truth_recovery.md"), headers, md_rows)


def aggregate_rows_with_std(
    rows: List[Dict[str, object]],
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, group_rows in grouped.items():
        base = {k: v for k, v in zip(group_keys, key)}
        base["num_runs"] = len(group_rows)
        for metric in metric_keys:
            vals = np.asarray([float(r[metric]) for r in group_rows], dtype=np.float64)
            base[metric] = float(np.mean(vals))
            base[f"{metric}_std"] = float(np.std(vals))
        out.append(base)
    return out


def save_predictor_summary(rows: List[Dict[str, object]], out_dir: str) -> None:
    metric_keys = ["best_val_acc", "test_acc", "test_auroc"]
    summary = aggregate_rows_with_std(
        rows,
        group_keys=["setting", "setting_label"],
        metric_keys=metric_keys,
    )
    fieldnames = ["setting", "setting_label", "num_runs"]
    for metric in metric_keys:
        fieldnames.extend([metric, f"{metric}_std"])
    write_csv(os.path.join(out_dir, "predictor_performance_summary.csv"), summary, fieldnames)


def save_recovery_summary(rows: List[Dict[str, object]], out_dir: str) -> None:
    metric_keys = [
        "cell_iou",
        "player_iou",
        "player_top1_hit",
        "player_top3_hit",
        "group_top1_hit",
        "segment_top1_hit",
        "avg_runtime_sec",
    ]
    summary = aggregate_rows_with_std(
        rows,
        group_keys=["setting", "setting_label", "method", "method_label"],
        metric_keys=metric_keys,
    )
    fieldnames = ["setting", "setting_label", "method", "method_label", "num_runs"]
    for metric in metric_keys:
        fieldnames.extend([metric, f"{metric}_std"])
    write_csv(os.path.join(out_dir, "ground_truth_recovery_summary.csv"), summary, fieldnames)


def merge_rows(
    existing_rows: List[Dict[str, object]],
    new_rows: List[Dict[str, object]],
    key_fields: Sequence[str],
) -> List[Dict[str, object]]:
    merged: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in existing_rows:
        key = tuple(row[k] for k in key_fields)
        merged[key] = row
    for row in new_rows:
        key = tuple(row[k] for k in key_fields)
        merged[key] = row
    return list(merged.values())



class TinyClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=1))


def smoke_test() -> None:
    set_global_seed(11)
    device = torch.device("cpu")
    cfg = SyntheticConfig(
        name="A",
        num_train=8,
        num_val=4,
        num_test=4,
        seq_len=20,
        num_features=4,
        num_groups=2,
        gt_segments_per_sample=4,
        explainer_segments_per_sample=4,
        hidden_dim=8,
        num_layers=1,
        epochs=1,
        batch_size=4,
        budget_calls=8,
        hsic_max_samples=20,
        eval_samples=1,
        num_seeds=1,
    )
    bundle = build_synthetic_bundle("A", cfg, seed=11)
    model = TinyClassifier(input_dim=cfg.num_features).to(device).eval()
    baseline_mean = compute_background_mean(bundle.train.X, max_samples=cfg.hsic_max_samples, seed=11)
    x_seq = bundle.test.X[0]
    y_true = int(bundle.test.y[0])
    groups = fixed_groups(cfg.num_features, cfg.num_groups)
    segments = fixed_segments(cfg.seq_len, cfg.explainer_segments_per_sample)
    for method in DEFAULT_METHODS:
        phi_cell, _elapsed = evaluate_method_on_sample(
            method=method,
            model=model,
            x_seq_np=x_seq,
            y_true=y_true,
            baseline_mean=baseline_mean,
            inferred_groups=groups,
            explainer_segments=segments,
            cfg=cfg,
            seed=11,
            device=device,
        )
        assert phi_cell.shape == x_seq.shape, (method, phi_cell.shape, x_seq.shape)
        assert np.all(np.isfinite(phi_cell)), method
    print("[SMOKE] synthetic_recovery_experiment.py synthetic smoke test passed.")


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test()
        return
    cfg = SyntheticConfig(
        name="unused",
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        budget_calls=args.budget_calls,
        hsic_max_samples=args.hsic_max_samples,
        eval_samples=args.eval_samples,
        num_seeds=args.num_seeds,
    )
    set_global_seed(args.seed)
    device = default_device()
    os.makedirs(args.out_dir, exist_ok=True)

    predictor_rows: List[Dict[str, object]] = []
    recovery_rows: List[Dict[str, object]] = []
    if args.append_existing:
        predictor_rows = read_csv_rows(os.path.join(args.out_dir, "predictor_performance.csv"))
        recovery_rows = read_csv_rows(os.path.join(args.out_dir, "ground_truth_recovery.csv"))

    settings_meta: Dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "settings": args.settings,
        "config": {
            "num_train": cfg.num_train,
            "num_val": cfg.num_val,
            "num_test": cfg.num_test,
            "seq_len": cfg.seq_len,
            "num_features": cfg.num_features,
            "num_groups": cfg.num_groups,
            "gt_segments_per_sample": cfg.gt_segments_per_sample,
            "explainer_segments_per_sample": cfg.explainer_segments_per_sample,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "epochs": cfg.epochs,
            "eval_samples": cfg.eval_samples,
            "budget_calls": cfg.budget_calls,
            "num_seeds": cfg.num_seeds,
        },
    }

    setting_iter = tqdm(args.settings, desc="Synthetic Settings", unit="setting")
    for setting in setting_iter:
        setting_iter.set_postfix_str(setting)
        setting_cfg = SyntheticConfig(**{**cfg.__dict__, "name": setting})
        setting_out_dir = os.path.join(args.out_dir, setting.lower())
        os.makedirs(setting_out_dir, exist_ok=True)

        seed_iter = tqdm(range(cfg.num_seeds), desc=f"{setting} seeds", unit="seed", leave=False)
        for seed_offset in seed_iter:
            run_seed = args.seed + 1000 * args.settings.index(setting) + 97 * seed_offset
            run_tag = f"seed{seed_offset}"
            run_out_dir = os.path.join(setting_out_dir, run_tag)
            os.makedirs(run_out_dir, exist_ok=True)
            bundle = build_synthetic_bundle(setting, setting_cfg, seed=run_seed)
            bundle_meta_path = os.path.join(run_out_dir, "bundle_meta.json")
            if args.reuse_models and os.path.exists(bundle_meta_path):
                bundle_meta = read_json(bundle_meta_path)
                inferred_groups = [
                    [int(feat) for feat in group]
                    for group in bundle_meta["inferred_groups"]
                ]
                explainer_segments = [
                    (int(seg[0]), int(seg[1]))
                    for seg in bundle_meta["explainer_segments"]
                ]
            else:
                inferred_groups = cluster_features_hsic(
                    bundle.train.X.reshape(-1, bundle.train.X.shape[-1]),
                    max_samples=cfg.hsic_max_samples,
                    seed=run_seed,
                )
                explainer_segments = fixed_segments(cfg.seq_len, cfg.explainer_segments_per_sample)

            model_path = os.path.join(run_out_dir, f"model_{bundle.config.name}.pt")
            if args.reuse_models:
                print(f"[SYNTH] Reusing predictor for setting={setting}, {run_tag}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing checkpoint for reuse: {model_path}")
                model = load_trained_model(setting_cfg, device=device, model_path=model_path)
            else:
                print(f"[SYNTH] Training predictor for setting={setting}, {run_tag}")
                model, perf = train_predictor(bundle, setting_cfg, device=device, out_dir=run_out_dir)
                perf["seed"] = run_tag
                predictor_rows.append(perf)

            print(f"[SYNTH] Evaluating explanation recovery for setting={setting}, {run_tag}")
            setting_recovery_rows = evaluate_recovery(
                bundle=bundle,
                model=model,
                cfg=setting_cfg,
                device=device,
                out_dir=run_out_dir,
                seed=run_seed,
                inferred_groups=inferred_groups,
                explainer_segments=explainer_segments,
                run_tag=run_tag,
                methods=args.methods,
            )
            recovery_rows.extend(setting_recovery_rows)
            recovery_rows = merge_rows([], recovery_rows, key_fields=["setting", "seed", "method"])
            save_recovery_tables(recovery_rows, args.out_dir)
            save_recovery_summary(recovery_rows, args.out_dir)

            with open(os.path.join(run_out_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "setting": setting,
                        "setting_label": SETTING_LABELS[setting],
                        "seed": run_tag,
                        "true_groups": bundle.true_groups,
                        "true_segments": bundle.true_segments,
                        "inferred_groups": inferred_groups,
                        "explainer_segments": explainer_segments,
                        "train_shape": list(bundle.train.X.shape),
                        "val_shape": list(bundle.val.X.shape),
                        "test_shape": list(bundle.test.X.shape),
                    },
                    f,
                    indent=2,
                )

    predictor_rows = merge_rows([], predictor_rows, key_fields=["setting", "seed"])
    recovery_rows = merge_rows([], recovery_rows, key_fields=["setting", "seed", "method"])

    if predictor_rows:
        save_predictor_tables(predictor_rows, args.out_dir)
        save_predictor_summary(predictor_rows, args.out_dir)
    save_recovery_tables(recovery_rows, args.out_dir)
    save_recovery_summary(recovery_rows, args.out_dir)
    with open(os.path.join(args.out_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(settings_meta, f, indent=2)

    print("[DONE] Synthetic ground-truth recovery results saved to:")
    print(" ", args.out_dir)


if __name__ == "__main__":
    main()
