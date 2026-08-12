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

<img width="5796" height="2520" alt="image" src="https://github.com/user-attachments/assets/8e2110f8-0a0d-4735-9b22-599edf7596da" />


## Abstract

Multivariate time-series models achieve strong predictive performance in healthcare, industry, energy, and finance, but how they combine cross-variable interactions with temporal dynamics remains unclear. SHapley Additive exPlanations (SHAP) has been widely used for interpretation. However, existing time-series variants typically treat the feature and time axes independently, fragmenting structural signals formed jointly by multiple variables over specific intervals. We propose GroupSegment-SHAP (GS-SHAP), which constructs explanatory units as group-segment players based on cross-variable dependence and distribution shifts over time, and then quantifies each unit's contribution via Shapley attribution. We evaluated GS-SHAP across four real-world domains: human activity recognition, power-system forecasting, medical signal analysis, and financial time series, and compared it with KernelSHAP, TimeSHAP, SequenceSHAP, WindowSHAP, and TSHAP. GS-SHAP achieves about 1.5× higher deletion-based faithfulness (ΔAUC) on average than the baselines. In synthetic evaluations with controlled ground-truth structures, GS-SHAP achieves 24.4% higher IoU-based recovery of multivariate-temporal patterns than the strongest baseline. In addition, GS-SHAP runs on average 3.38× faster than TimeSHAP across approximation budgets in power-system forecasting, showing that it can simultaneously achieve high explanatory faithfulness and computational efficiency. In a financial case study, GS-SHAP identifies interpretable multivariate-temporal interactions among key market variables across market regimes, highlighting its potential utility for risk-aware investment analysis.

<br>

## Motivation

- Existing SHAP-based time-series explainers often define players along a single axis, such as individual cells, time steps, fixed windows, or temporal subsequences.

- These player definitions can fragment multivariate-temporal evidence that arises jointly from multiple variables over specific intervals.

- In many practical settings, the most meaningful predictive evidence is formed by groups of variables changing together within specific temporal regimes.

- GS-SHAP addresses this player-space mismatch by constructing explanatory units that jointly reflect cross-variable dependence and temporal distribution shifts.

<br>

## Contribution

- **Group-segment players.** Group-segment players are introduced as a new Shapley player space that addresses player-space mismatch by representing structurally coherent variable-time regions in multivariate time series.

- **General player-construction and attribution framework.** GS-SHAP reduces attribution fragmentation caused by feature-only, time-only, or fixed-window player definitions.

- **Faithful, stable, and computationally efficient explanations.** GS-SHAP provides reliable attribution across heterogeneous multivariate time-series domains and controlled synthetic benchmarks.

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
├── preprocessing/          # Dataset preprocessing
├── scripts/                # Main experiment scripts
├── DEPENDENCIES.md
├── MANIFEST.md
└── README.md
```

---

### 3. Pipeline Overview

#### 3.1 HSIC-Based Feature Grouping

GS-SHAP first partitions the variable space into feature groups based on nonlinear cross-variable dependence. The implementation constructs an HSIC affinity matrix, determines the number of groups using the eigengap criterion, and applies spectral clustering.

The HSIC affinity is estimated once from the training set and reused across explanation runs.

#### 3.2 MMD-Based Temporal Segmentation

For each feature group, GS-SHAP detects distributional changes using Maximum Mean Discrepancy (MMD) and recursively partitions the time axis into group-specific temporal segments.

The implementation uses a permutation-calibrated split threshold and dataset-specific minimum segment lengths.

#### 3.3 Group-Segment Player Construction

Each feature group is combined with its own temporal segments to define group-segment players of the form

```text
G_k × S_j^(k)
```

The resulting players form a non-overlapping partition of the input grid.

#### 3.4 GroupSegment-SHAP Attribution

GS-SHAP estimates each player's marginal contribution using permutation-based Shapley approximation.

Mean replacement is used as the default masking rule, with feature-wise means computed from the background set.

---

### 4. Experimental Setup

GS-SHAP is evaluated on four multivariate time-series datasets spanning different application domains.

| Dataset | Domain | Prediction Task | Window Size |
|---|---|---|---:|
| HAR | Human activity recognition | Classification | 96 |
| ETTm1 | Power-system forecasting | Regression | 128 |
| PTB-XL | Medical signal analysis | Classification | 1000 |
| S&P500 | Financial time series | Regression | 20 |

The main experiments use a fixed BiLSTM predictor across the four datasets to isolate the effect of explainer design.

GS-SHAP is compared with:

- KernelSHAP
- TimeSHAP
- SequenceSHAP
- WindowSHAP
- TSHAP

Additional backbone evaluations are conducted with Transformer and Mamba models.

---

### 5. Main Experiments

#### 5.1 Faithfulness Evaluation

Faithfulness is evaluated using a deletion protocol that progressively masks high-attribution cells and measures the resulting increase in prediction loss.

| Method | HAR | ETTm1 | PTB-XL | S&P500 |
|---|---:|---:|---:|---:|
| KernelSHAP | 0.523 | 10.344 | 0.066 | 1.18×10⁻⁵ |
| TimeSHAP | 2.294 | 14.867 | 0.098 | 2.41×10⁻⁵ |
| SequenceSHAP | 3.453 | 22.208 | 0.132 | 3.70×10⁻⁵ |
| WindowSHAP | 1.912 | 17.344 | 0.087 | 1.62×10⁻⁵ |
| TSHAP | 3.089 | 24.411 | 0.087 | 3.24×10⁻⁵ |
| **GS-SHAP** | **3.955** | **26.524** | **0.171** | **5.94×10⁻⁵** |

GS-SHAP achieves a mean ΔAUC of 7.66, about 52% higher than the baseline average of 5.05.

#### 5.2 Synthetic Ground-Truth Recovery

Synthetic multivariate time series with controlled ground-truth structures are used to evaluate structural recovery.

| Method | Avg. Cell IoU | Avg. Player IoU |
|---|---:|---:|
| KernelSHAP | 0.080 | 0.293 |
| TimeSHAP | 0.081 | 0.109 |
| SequenceSHAP | 0.324 | 0.503 |
| WindowSHAP | 0.085 | 0.115 |
| TSHAP | 0.085 | 0.118 |
| **GS-SHAP** | **0.403** | **0.538** |

GS-SHAP improves the average Cell IoU by 24.4% over the second-best method, SequenceSHAP.

#### 5.3 Feature Grouping Analysis

HSIC grouping is compared with Pearson-correlation grouping, random grouping, and no grouping.

Component-level ablation additionally evaluates:

- GS-SHAP: HSIC + MMD
- w/o HSIC: Singleton + MMD
- w/o MMD: HSIC + count-matched fixed windows
- w/o HSIC, MMD: Singleton + count-matched fixed windows

A controlled dependency-shift experiment also compares static HSIC grouping with segment-wise dynamic regrouping.

#### 5.4 Robustness and Sensitivity

The paper evaluates attribution stability under different background sets and examines sensitivity to:

- minimum segment length
- mean masking
- zero masking
- noise masking

#### 5.5 Computational Efficiency

Runtime is reported under matched approximation budgets.

On ETTm1, GS-SHAP runs on average 3.38× faster than TimeSHAP across the evaluated approximation budgets.

The player-space analysis shows an average 96.9% reduction relative to the original cell-level input space.

---

### 6. S&P500 Case Study

The paper analyzes GS-SHAP explanations under high-volatility and stable market regimes for next-day S&P500 return prediction.

The case study shows that GS-SHAP localizes model contributions to specific combinations of feature groups and temporal intervals, revealing regime-dependent shifts in the model's information sources.

---

### 7. Smoke Test

```bash
python scripts/deletion_experiment.py --smoke
python scripts/synthetic_recovery_experiment.py --smoke
```

---

### 8. Data

The experiments use:

- UCI Human Activity Recognition (HAR)
- ETTm1
- PTB-XL
- S&P500 financial time-series data

Dataset preprocessing utilities are provided under `preprocessing/`.

Original datasets should be obtained from their respective data providers.

---

### 9. Scope of the Released Code

This repository contains the implementation for the main computational components reported in the paper, including:

- HSIC-based feature grouping
- MMD-based temporal segmentation
- group-segment player construction
- Shapley attribution
- dataset preprocessing
- deletion-based faithfulness evaluation
- synthetic recovery evaluation
- grouping and ablation experiments
- sensitivity analyses
- backbone generalization
- runtime and player-space analyses

<br>

## Citation

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
