# GroupSegment-SHAP: Shapley Value Explanations with Group-Segment Players for Multivariate Time Series

<p align="center">
  <b>Jinwoong Kim<sup>1</sup> · Sangjin Park<sup>1*</sup></b><br>
  <sup>1</sup>Graduate School of Industrial Data Engineering, Hanyang University, Seoul, Republic of Korea<br>
  <sup>*</sup>Corresponding author
</p>

****
[![Paper](https://img.shields.io/badge/Paper-CIKM_2026-red?style=for-the-badge)](#citation)
[![Conference](https://img.shields.io/badge/Conference-CIKM_'26-blue?style=for-the-badge)](https://cikm2026.diag.uniroma1.it/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.06114-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2601.06114)

<!-- Main figure will be inserted here -->

## Abstract

Multivariate time-series models achieve strong predictive performance in healthcare, industry, energy, and finance, but how they combine cross-variable interactions with temporal dynamics remains unclear. Existing time-series SHAP variants typically treat the feature and time axes independently, fragmenting structural signals formed jointly by multiple variables over specific intervals. We propose GroupSegment-SHAP (GS-SHAP), which constructs explanatory units as group-segment players based on cross-variable dependence and distribution shifts over time and quantifies each unit's contribution via Shapley attribution. We evaluate GS-SHAP across four real-world domains: human activity recognition, power-system forecasting, medical signal analysis, and financial time series. Compared with KernelSHAP, TimeSHAP, SequenceSHAP, WindowSHAP, and TSHAP, GS-SHAP achieves about 1.5× higher deletion-based faithfulness on average. In controlled synthetic evaluations, GS-SHAP achieves 24.4% higher IoU-based recovery than the strongest baseline. GS-SHAP also runs on average 3.38× faster than TimeSHAP across approximation budgets in power-system forecasting. These results demonstrate that group-segment players provide a faithful and computationally efficient explanation space for multivariate time-series models.

<br>

## Motivation

- Multivariate time-series predictions often depend on **multiple variables evolving jointly over specific temporal intervals**, rather than isolated features or individual time points.

- Existing SHAP-based time-series explainers typically define players as individual cells, time steps, subsequences, or fixed temporal windows, which can fragment coupled multivariate-temporal patterns.

- A suitable explanatory unit should jointly account for **cross-variable dependence** and **temporal distributional changes**.

- We introduce **GS-SHAP**, which defines each Shapley player as the intersection of a dependency-based feature group and a group-specific temporal segment.

<br>

## Contribution

- **Group-segment explanatory units.** HSIC-based nonlinear dependency grouping and group-wise MMD segmentation define structurally coherent multivariate-temporal players.

- **Joint feature-time attribution.** GS-SHAP directly estimates Shapley values over group-segment players, reducing attribution fragmentation caused by feature-only, time-only, or fixed-window player definitions.

- **Empirical faithfulness and structural recovery.** Across HAR, ETTm1, PTB-XL, and S&P500, GS-SHAP achieves the highest deletion-based faithfulness among the compared explainers and improves average synthetic Cell IoU by 24.4% over the strongest baseline.

- **Computational efficiency and robustness.** GS-SHAP substantially reduces the player space and runs on average 3.38× faster than TimeSHAP on ETTm1 while remaining robust across masking baselines, segmentation settings, and predictive backbones.

> **Note:** HSIC measures statistical dependence rather than causality. The resulting feature groups should therefore be interpreted as dependence-based explanatory units, not causal groups.

<br>

## Build & Run Guide

### 1. Environment Setup

```bash
git clone https://github.com/jirungx/GroupSegment-SHAP.git
cd GroupSegment-SHAP

conda create -n gsshap python=3.10 -y
conda activate gsshap
```

Install the required dependencies following `DEPENDENCIES.md`.

---

### 2. Project Structure

```text
GroupSegment-SHAP/
├── gsshap/                 # Core GS-SHAP implementation
├── preprocessing/          # Dataset preprocessing for HAR, ETTm1, PTB-XL, and S&P500
├── scripts/                # Main executable experiments and evaluations
├── DEPENDENCIES.md         # Environment and dependency information
├── MANIFEST.md             # Repository file manifest
└── README.md
```

---

### 3. Pipeline Overview

#### 3.1 Data Preprocessing

Dataset preparation utilities are provided under `preprocessing/` for the four datasets used in the paper:

- HAR
- ETTm1
- PTB-XL
- S&P500

The preprocessing pipeline constructs the fixed-length model inputs and dataset-specific variables used in the experiments.

#### 3.2 GS-SHAP Attribution

The core implementation under `gsshap/` follows the four-stage procedure described in the paper:

1. HSIC-based feature grouping
2. Group-wise MMD temporal segmentation
3. Group-segment player construction
4. Permutation-based Shapley attribution

HSIC grouping uses up to 3,000 pooled training observations, selects the number of groups using the eigengap criterion, and applies spectral clustering to the HSIC affinity matrix.

MMD segmentation is performed independently for each feature group with a permutation-calibrated split threshold.

Default minimum segment lengths are:

| Dataset | Minimum segment length |
|---|---:|
| HAR | 10 |
| ETTm1 | 13 |
| PTB-XL | 100 |
| S&P500 | 4 |

Shapley attribution explains target-class model outputs for classification and scalar predictions for regression. Mean replacement based on a fixed sampled training background set is used as the default masking strategy.

#### 3.3 Deletion-Based Faithfulness

The deletion evaluation projects GS-SHAP and baseline attributions onto a common cell-level map and progressively masks the highest-importance cells.

The paper compares:

- KernelSHAP
- TimeSHAP
- SequenceSHAP
- WindowSHAP
- TSHAP
- GS-SHAP

under matched prediction models, samples, background sets, masking rules, and model-query budgets whenever applicable.

#### 3.4 Synthetic Ground-Truth Recovery

Controlled synthetic experiments evaluate whether each explanation method recovers predefined multivariate-temporal target structures.

Three settings are considered:

- **Single:** one target-generating group-segment player
- **Two:** two target-generating group-segment players
- **Distractor:** correlated but target-irrelevant group-segment patterns

Recovery is evaluated using **Cell IoU** and **Player IoU**.

Because the synthetic ground-truth patterns are constructed as group-segment structures, this evaluation is structurally aligned with GS-SHAP and should be interpreted as controlled evidence of structural recovery rather than an unbiased comparison across all possible attribution structures.

#### 3.5 Additional Analyses

The released experiments also cover:

- HSIC vs. Pearson/random/single-feature grouping
- HSIC–MMD component ablation
- Static vs. dynamic HSIC grouping
- Minimum-segment-length sensitivity
- Masking-baseline sensitivity
- Transformer backbone evaluation
- Mamba backbone evaluation
- Player-space reduction
- Runtime analysis

#### 3.6 Smoke Test

Lightweight smoke tests are provided to validate the experiment pipeline.

```bash
python scripts/deletion_experiment.py --smoke
python scripts/synthetic_recovery_experiment.py --smoke
```

---

### 4. Data

The study evaluates GS-SHAP on four multivariate time-series datasets.

| Dataset | Domain | Prediction Task | Window Size |
|---|---|---|---:|
| HAR | Human activity recognition | Classification | 96 |
| ETTm1 | Power-system forecasting | Regression | 128 |
| PTB-XL | ECG analysis | Classification | 1000 |
| S&P500 | Financial time series | Regression | 20 |

The repository provides preprocessing utilities for constructing the inputs used in the paper.

Original datasets should be obtained from their respective data providers. Dataset files are not redistributed where separate access or licensing conditions apply.

---

### 5. Main Results

#### Deletion-Based Faithfulness

| Method | HAR | ETTm1 | PTB-XL | S&P500 |
|---|---:|---:|---:|---:|
| KernelSHAP | 0.523 | 10.344 | 0.066 | 1.18×10⁻⁵ |
| TimeSHAP | 2.294 | 14.867 | 0.098 | 2.41×10⁻⁵ |
| SequenceSHAP | 3.453 | 22.208 | 0.132 | 3.70×10⁻⁵ |
| WindowSHAP | 1.912 | 17.344 | 0.087 | 1.62×10⁻⁵ |
| TSHAP | 3.089 | 24.411 | 0.087 | 3.24×10⁻⁵ |
| **GS-SHAP** | **3.955** | **26.524** | **0.171** | **5.94×10⁻⁵** |

GS-SHAP achieves approximately **1.5× higher deletion-based faithfulness on average** than the compared baselines.

#### Synthetic Ground-Truth Recovery

| Method | Avg. Cell IoU | Avg. Player IoU |
|---|---:|---:|
| KernelSHAP | 0.080 | 0.293 |
| TimeSHAP | 0.081 | 0.109 |
| SequenceSHAP | 0.324 | 0.503 |
| WindowSHAP | 0.085 | 0.115 |
| TSHAP | 0.085 | 0.118 |
| **GS-SHAP** | **0.403** | **0.538** |

GS-SHAP improves average Cell IoU by **24.4%** over the strongest baseline.

#### Runtime

Across approximation budgets on ETTm1, GS-SHAP runs on average **3.38× faster than TimeSHAP**.

The group-segment construction reduces the original cell-level player space by an average of **96.9%** across the four datasets.

---

### 6. Scope of the Released Code

This repository contains the code used for the main computational components reported in the paper:

- dataset preprocessing
- HSIC-based feature grouping
- MMD-based temporal segmentation
- group-segment player construction
- permutation-based Shapley attribution
- deletion-based faithfulness evaluation
- synthetic ground-truth recovery
- grouping analyses
- component ablations
- sensitivity analyses
- backbone generalization
- runtime evaluation
- player-space analysis

Figure-rendering and camera-ready manuscript post-processing files are not part of the core released implementation.

---

<br>

## Citation

If you use this code or GS-SHAP in your research, please cite:

```bibtex
@inproceedings{kim2026groupsegment,
  title     = {GroupSegment-SHAP: Shapley Value Explanations with Group-Segment Players for Multivariate Time Series},
  author    = {Kim, Jinwoong and Park, Sangjin},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026}
}
```

Preprint:

> Jinwoong Kim and Sangjin Park. *GroupSegment-SHAP: Shapley Value Explanations with Group-Segment Players for Multivariate Time Series.* arXiv:2601.06114 (2026).

<br>

## Acknowledgements

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT)(No. RS 2025-00554384), and the Technology Development Program (No.RS 202400513926) funded by the Ministry of SMEs and Startups(MSS, Korea).

## License

Please refer to the repository license and the licenses or terms of use of the individual datasets and third-party dependencies before redistribution or reuse.
