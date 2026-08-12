# GS-SHAP Code

Clean code package for GS-SHAP. The implementation follows the paper's player definition: HSIC feature groups are segmented independently by MMD, producing players of the form `G_k x S_j^(k)`.

```text
gsshap/          Core implementation: HSIC grouping, group-wise MMD segmentation, GS-SHAP attribution, deletion utilities, models, datasets.
scripts/         Main executable experiments.
preprocessing/   Dataset preparation CLI for HAR, ETTm1, PTB-XL, and S&P500.
```

## Implementation Defaults

- HSIC grouping uses up to 3,000 pooled training observations, selects the number of groups by the eigengap criterion from the HSIC affinity matrix, and applies one spectral-clustering pass. No dataset-level fixed group count is imposed.
- MMD segmentation is run separately for each feature group with a permutation-calibrated split threshold.
- Default minimum segment lengths follow the paper: HAR 10, ETTm1 13, PTB-XL 100, and S&P500 4.
- The BiLSTM backbone defaults to 2 layers, hidden size 64, and dropout 0.2, matching the implementation settings reported in the paper.
- ETTm1 preprocessing uses a 128-step input window and a 4-step horizon for 1-hour-ahead prediction on 15-minute data.
- HAR preprocessing crops the raw 128-step inertial windows to the paper-aligned 96-step input window.
- Deletion evaluation projects GS-SHAP player attributions to a common `T x D` cell map and uses mean replacement based on a fixed sampled training background set.
- Shapley attribution explains the model output `f(X)`: target-class logits for classification and scalar predictions for regression. Deletion faithfulness is then evaluated by the corresponding prediction-loss increase.

## Smoke Test

```bash
python scripts/deletion_experiment.py --smoke
python scripts/synthetic_recovery_experiment.py --smoke
```

## Main Deletion Experiments

```bash
python scripts/deletion_experiment.py --datasets ettm1 --num-samples 200
python scripts/deletion_experiment.py --datasets har ettm1 ptbxl sp500
```

## Synthetic Ground-Truth Recovery

```bash
python scripts/synthetic_recovery_experiment.py
```

Fast development run:

```bash
python scripts/synthetic_recovery_experiment.py --settings A --num-train 200 --num-val 50 --num-test 50 --eval-samples 5 --num-seeds 1 --epochs 1 --budget-calls 100
```

## Preprocessing

```bash
python preprocessing/prepare_dataset.py har
python preprocessing/prepare_dataset.py ettm1
python preprocessing/prepare_dataset.py ptbxl
python preprocessing/prepare_dataset.py sp500
```

Datasets and trained checkpoints are expected under `data/` and `models/` in the package root. They are not included in this repository.

## Reproducibility

This repository provides the core implementation of GS-SHAP, including HSIC-based feature grouping, group-wise MMD-based temporal segmentation, group-segment player construction, Shapley attribution, deletion-based faithfulness evaluation, and synthetic ground-truth recovery evaluation.
