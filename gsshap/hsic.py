import math
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from tqdm.auto import tqdm


def rbf_kernel_1d(x: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    x: (N,)
    1D RBF kernel matrix K_ij = exp(- (x_i - x_j)^2 / (2 sigma^2))
    """
    x = x.reshape(-1, 1).astype(np.float32)
    dists = (x - x.T) ** 2
    if sigma is None:
        med = np.median(dists)
        sigma = math.sqrt(med + 1e-8)
    return np.exp(-dists / (2 * sigma**2 + 1e-8)).astype(np.float32)


def hsic(X: np.ndarray, Y: np.ndarray) -> float:
    """
    X, Y: (N,)
    HSIC(X, Y) with RBF kernel
    """
    n = X.shape[0]
    K = rbf_kernel_1d(X)
    L = rbf_kernel_1d(Y)
    H = np.eye(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
    HKH = H @ K @ H
    HLH = H @ L @ H
    return float(np.trace(HKH @ HLH) / ((n - 1) ** 2))


def _resolve_torch_device(device: torch.device | str | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def rbf_kernel_1d_torch(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """
    Torch version of the 1D RBF kernel for GPU-backed HSIC.
    """
    x = x.reshape(-1, 1).float()
    dists = (x - x.transpose(0, 1)) ** 2
    if sigma is None:
        med = torch.median(dists)
        sigma = math.sqrt(float(med.item()) + 1e-8)
    return torch.exp(-dists / (2.0 * sigma**2 + 1e-8)).float()


def hsic_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device | str | None = None,
) -> float:
    """
    Torch/GPU-capable HSIC(X, Y) with RBF kernel.
    """
    dev = _resolve_torch_device(device)
    X = X.to(dev, dtype=torch.float32)
    Y = Y.to(dev, dtype=torch.float32)

    n = int(X.shape[0])
    if n < 2:
        return 0.0

    with torch.no_grad():
        K = rbf_kernel_1d_torch(X)
        L = rbf_kernel_1d_torch(Y)
        H = torch.eye(n, device=dev, dtype=torch.float32)
        H = H - torch.ones((n, n), device=dev, dtype=torch.float32) / float(n)
        HKH = H @ K @ H
        HLH = H @ L @ H
        out = torch.trace(HKH @ HLH) / float((n - 1) ** 2)
    return float(out.item())


def build_hsic_matrix(
    X_all: np.ndarray,
    max_samples: int = 5000,
    use_torch: bool = False,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    X_all: (N, D)
    """
    N, D = X_all.shape
    if N > max_samples:
        idx = np.random.choice(N, size=max_samples, replace=False)
        X = X_all[idx]
        print(f"[HSIC] Using {max_samples} samples out of {N} for HSIC matrix")
    else:
        X = X_all

    hsic_mat = np.zeros((D, D), dtype=np.float32)
    X_torch = None
    torch_device = _resolve_torch_device(device) if use_torch else None
    if use_torch:
        X_torch = torch.as_tensor(X, dtype=torch.float32, device=torch_device)
        print(f"[HSIC] backend=torch device={torch_device} shape=({X.shape[0]}, {D})")
    else:
        print(f"[HSIC] backend=numpy device=cpu shape=({X.shape[0]}, {D})")

    total_pairs = D * (D + 1) // 2
    pbar = tqdm(total=total_pairs, desc="HSIC pairs", unit="pair", leave=False)
    for i in range(D):
        for j in range(i, D):
            if use_torch and X_torch is not None:
                val = hsic_torch(X_torch[:, i], X_torch[:, j], device=torch_device)
            else:
                val = hsic(X[:, i], X[:, j])
            hsic_mat[i, j] = val
            hsic_mat[j, i] = val
            pbar.update(1)
        pbar.set_postfix_str(f"row={i+1}/{D}")
    pbar.close()
    return hsic_mat


def estimate_num_groups_eigengap(hsic_mat: np.ndarray, max_k: Optional[int] = None) -> int:
    """Estimate the number of feature groups with the eigengap heuristic.

    When max_k is None, all valid K values are considered from the HSIC
    affinity matrix, matching the manuscript description that K is selected
    by eigengap rather than fixed as a dataset hyperparameter.
    """
    D = hsic_mat.shape[0]
    if D <= 2:
        return D

    W = hsic_mat.copy()
    np.fill_diagonal(W, 0.0)

    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-8))
    L = np.eye(D) - D_inv_sqrt @ W @ D_inv_sqrt  # normalized Laplacian

    vals, _ = np.linalg.eigh(L)
    vals = np.sort(vals)

    max_k_eff = D - 1 if max_k is None else min(int(max_k), D - 1)
    k_candidates = max_k_eff + 1
    gaps = np.diff(vals[:k_candidates])
    k_opt = int(np.argmax(gaps) + 1)

    if k_opt < 2:
        k_opt = 2

    print(f"[HSIC] eigengap-based num_groups = {k_opt} (D={D})")
    return k_opt


def spectral_cluster_hsic_affinity(
    hsic_mat: np.ndarray,
    max_k: Optional[int] = None,
    seed: int = 42,
) -> List[List[int]]:
    """
    Cluster variables exactly as described in the paper:
      1) estimate the number of groups K with the eigengap criterion,
      2) apply spectral clustering on the HSIC affinity matrix.

    By default, K is not fixed by a dataset-level max-group setting; the
    eigengap is evaluated over the valid spectrum of the affinity matrix. The
    diagonal is removed before graph construction so that self-HSIC values do
    not dominate inter-variable affinity.
    """
    W = np.asarray(hsic_mat, dtype=np.float32).copy()
    D = W.shape[0]
    if W.shape != (D, D):
        raise ValueError(f"HSIC affinity must be square, got {W.shape}")
    if D == 0:
        return []
    if D == 1:
        return [[0]]

    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, 0.0)

    if float(W.sum()) <= 1e-12:
        print("[HSIC] zero affinity graph; falling back to singleton groups")
        return [[i] for i in range(D)]

    num_groups = estimate_num_groups_eigengap(W, max_k=max_k)
    num_groups = max(1, min(int(num_groups), D))

    if num_groups == 1:
        return [list(range(D))]
    if num_groups == D:
        return [[i] for i in range(D)]

    clustering = SpectralClustering(
        n_clusters=num_groups,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    )
    labels = clustering.fit_predict(W)

    groups: List[List[int]] = []
    for g in range(num_groups):
        idx = np.where(labels == g)[0].tolist()
        if idx:
            groups.append(sorted(idx))

    groups = sorted(groups, key=lambda x: (min(x), len(x)))
    print(f"[HSIC] spectral-clustering groups = {groups}")
    return groups


def cluster_features_hsic(
    X_all: np.ndarray,
    max_samples: int = 3000,
    max_k: Optional[int] = None,
    seed: int = 42,
    use_torch: bool = False,
    device: torch.device | str | None = None,
    precomputed_hsic: np.ndarray | None = None,
    max_depth: int | None = None,
    min_avg_hsic: float | None = None,
) -> List[List[int]]:
    """Build HSIC-based feature groups by eigengap and spectral clustering."""
    _, D = X_all.shape

    if precomputed_hsic is not None:
        hsic_mat_full = np.asarray(precomputed_hsic, dtype=np.float32)
        if hsic_mat_full.shape != (D, D):
            raise ValueError(
                f"precomputed_hsic shape mismatch: expected {(D, D)}, got {hsic_mat_full.shape}"
            )
        print(f"[HSIC] Reusing precomputed HSIC matrix for D={D} features")
    else:
        hsic_mat_full = build_hsic_matrix(
            X_all,
            max_samples=max_samples,
            use_torch=use_torch,
            device=device,
        )
        print(f"[HSIC] Full HSIC matrix built for D={D} features")

    # Follow the manuscript: eigengap-selected K, then one spectral-clustering pass.
    groups = spectral_cluster_hsic_affinity(hsic_mat_full, max_k=max_k, seed=seed)
    return [sorted(g) for g in groups]
