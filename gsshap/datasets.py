from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset


class NpzSeqDataset(Dataset):
    """Dataset wrapper for sequence NPZ files with arrays `X`, `y`, and optional `dates`."""

    def __init__(self, npz_path: str, dates: Optional[np.ndarray] = None):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"]
        self.dates = data["dates"] if "dates" in data.files else dates

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def has_dates(self) -> bool:
        return self.dates is not None

    def get_date(self, idx: int) -> Any:
        if self.dates is None:
            raise RuntimeError("This dataset does not include dates.")
        return self.dates[idx]


def split_train_val(full_dataset: Dataset, val_ratio: float, seed: int = 42):
    """Return deterministic train/validation subsets."""
    n = len(full_dataset)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_val = int(n * val_ratio)
    return Subset(full_dataset, indices[n_val:]), Subset(full_dataset, indices[:n_val])


def _load_optional_dates(base_dir: str, prefix: str, split: str) -> Optional[np.ndarray]:
    npy_path = os.path.join(base_dir, f"{prefix}_{split}_dates.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True)
    csv_path = os.path.join(base_dir, f"{prefix}_{split}_dates.csv")
    if os.path.exists(csv_path):
        return np.loadtxt(csv_path, dtype=str, delimiter=",")
    return None


def _make_loaders(full_train: NpzSeqDataset, test_ds: NpzSeqDataset, batch_size: int, val_ratio: float):
    train_ds, val_ds = split_train_val(full_train, val_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return full_train, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def load_har_dataloaders(data_dir: str, batch_size: int = 128, val_ratio: float = 0.1):
    train_npz = os.path.join(data_dir, "HAR", "har_train.npz")
    test_npz = os.path.join(data_dir, "HAR", "har_test.npz")
    full_train = NpzSeqDataset(train_npz)
    test_ds = NpzSeqDataset(test_npz)
    full_train.y = full_train.y.astype(np.int64)
    test_ds.y = test_ds.y.astype(np.int64)
    return _make_loaders(full_train, test_ds, batch_size, val_ratio)


def load_ettm1_dataloaders(data_dir: str, batch_size: int = 128, val_ratio: float = 0.1):
    train_npz = os.path.join(data_dir, "ETTm1", "ett_train.npz")
    test_npz = os.path.join(data_dir, "ETTm1", "ett_test.npz")
    full_train = NpzSeqDataset(train_npz)
    test_ds = NpzSeqDataset(test_npz)
    full_train.y = full_train.y.astype(np.float32)
    test_ds.y = test_ds.y.astype(np.float32)
    return _make_loaders(full_train, test_ds, batch_size, val_ratio)


def load_ptbxl_dataloaders(data_dir: str, batch_size: int = 128, val_ratio: float = 0.1):
    train_npz = os.path.join(data_dir, "PTBXL", "ptbxl_train.npz")
    test_npz = os.path.join(data_dir, "PTBXL", "ptbxl_test.npz")
    full_train = NpzSeqDataset(train_npz)
    test_ds = NpzSeqDataset(test_npz)
    full_train.y = full_train.y.astype(np.int64)
    test_ds.y = test_ds.y.astype(np.int64)
    return _make_loaders(full_train, test_ds, batch_size, val_ratio)


def load_sp500_dataloaders(
    data_dir: str,
    horizon_index: int = 0,
    batch_size: int = 128,
    val_ratio: float = 0.1,
):
    sp500_dir = os.path.join(data_dir, "SP500")
    train_dates = _load_optional_dates(sp500_dir, "sp500", "train")
    test_dates = _load_optional_dates(sp500_dir, "sp500", "test")
    full_train = NpzSeqDataset(os.path.join(sp500_dir, "sp500_train.npz"), dates=train_dates)
    test_ds = NpzSeqDataset(os.path.join(sp500_dir, "sp500_test.npz"), dates=test_dates)
    full_train.y = full_train.y[:, horizon_index].astype(np.float32)
    test_ds.y = test_ds.y[:, horizon_index].astype(np.float32)
    return _make_loaders(full_train, test_ds, batch_size, val_ratio)
