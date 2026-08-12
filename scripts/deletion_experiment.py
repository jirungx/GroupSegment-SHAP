#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from tqdm.auto import tqdm

from gsshap.evaluation import (
    DATASETS,
    DEFAULT_METHODS,
    FRACTIONS,
    build_feature_groups,
    build_model,
    compute_background_mean_from_sample,
    default_device,
    empty_score_buffers,
    evaluate_sample,
    load_bundle,
    set_seed,
    store_scores,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GS-SHAP deletion faithfulness experiments.")
    p.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["ettm1"])
    p.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--budget-calls", type=int, default=1500)
    p.add_argument("--min-perms", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--output-dir", type=str, default="results/deletion")
    p.add_argument("--smoke", action="store_true", help="Run a tiny synthetic smoke test without data files.")
    return p.parse_args()


class TinyRegressor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2), keepdim=True).view(x.shape[0], 1)


def smoke_test() -> None:
    from gsshap.evaluation import DATASETS

    set_seed(7)
    spec = DATASETS["ettm1"]
    spec = type(spec)(**{**spec.__dict__, "max_segments": 2, "min_seg_len": 2})
    x = np.random.randn(12, 4).astype(np.float32)
    y = np.float32(0.1)
    groups = [[0, 1], [2, 3]]
    baseline = np.zeros((4,), dtype=np.float32)
    scores = evaluate_sample(
        model=TinyRegressor(),
        spec=spec,
        x_seq=x,
        y_value=y,
        groups=groups,
        baseline_mean=baseline,
        methods=DEFAULT_METHODS,
        fractions=[0.0, 0.5],
        budget_calls=24,
        min_perms=1,
        seed=7,
        device=torch.device("cpu"),
    )
    assert set(scores) == {"joint"}
    assert 0.0 in scores["joint"] and 0.5 in scores["joint"]
    print("[SMOKE] deletion_experiment.py synthetic smoke test passed.")


def run_dataset(args: argparse.Namespace, dataset_key: str) -> None:
    spec = DATASETS[dataset_key]
    device = default_device()
    set_seed(args.seed)

    full_train, _train_ds, _val_ds, test_ds, *_ = load_bundle(spec, args.data_dir)
    input_dim = int(full_train.X.shape[-1])
    output_dim = int(len(np.unique(full_train.y))) if spec.task == "clf" else 1
    model = build_model(spec, input_dim, output_dim, args.model_dir, device)
    groups = build_feature_groups(full_train, spec, args.seed)
    baseline_mean = compute_background_mean_from_sample(
        full_train.X,
        max_samples=spec.hsic_max_samples,
        seed=args.seed,
    )

    n = min(args.num_samples, len(test_ds))
    indices = np.random.RandomState(args.seed).choice(len(test_ds), size=n, replace=False)
    buffers = empty_score_buffers(args.methods, FRACTIONS, n)

    print(f"[RUN] {spec.label}: samples={n}, methods={args.methods}, groups={groups}")
    for pos, idx in enumerate(tqdm(indices, desc=f"{spec.label} deletion", unit="sample")):
        x_seq, y_value = test_ds[int(idx)]
        scores = evaluate_sample(
            model=model,
            spec=spec,
            x_seq=x_seq,
            y_value=y_value,
            groups=groups,
            baseline_mean=baseline_mean,
            methods=args.methods,
            fractions=FRACTIONS,
            budget_calls=args.budget_calls,
            min_perms=args.min_perms,
            seed=args.seed + int(idx),
            device=device,
        )
        for method, score_map in scores.items():
            store_scores(buffers[method], FRACTIONS, score_map, pos)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_key}_deletion_scores.npz"
    payload = {"fractions": np.array(FRACTIONS), "indices": indices, "groups": np.array(groups, dtype=object)}
    payload.update({f"scores_{m}": buffers[m] for m in args.methods})
    np.savez_compressed(out_path, **payload)
    print(f"[DONE] Saved {out_path}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        smoke_test()
        return
    for dataset in args.datasets:
        run_dataset(args, dataset)


if __name__ == "__main__":
    main()
