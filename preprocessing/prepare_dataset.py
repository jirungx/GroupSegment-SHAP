#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"



def prepare_har() -> None:
    dataset_dir = DATA_DIR / "HAR"
    raw_root_candidates = [
        dataset_dir / "UCI HAR Dataset",
        dataset_dir,
    ]
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def find_raw_root() -> Path:
        for root in raw_root_candidates:
            if (root / "train" / "Inertial Signals").exists():
                return root
        tried = ", ".join(str(x) for x in raw_root_candidates)
        raise FileNotFoundError(
            "Missing UCI HAR raw files. Expected train/test/Inertial Signals under one of: " + tried
        )

    def load_split(raw_root: Path, split: str):
        signal_dir = raw_root / split / "Inertial Signals"
        arrays = []
        for name in signal_names:
            path = signal_dir / f"{name}_{split}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Missing HAR signal file: {path}")
            arrays.append(np.loadtxt(path, dtype=np.float32))
        X = np.stack(arrays, axis=-1).astype(np.float32)
        if X.shape[1] < t_window:
            raise ValueError(f"HAR sequence length {X.shape[1]} is shorter than t_window={t_window}.")
        X = X[:, :t_window, :]
        y_path = raw_root / split / f"y_{split}.txt"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing HAR label file: {y_path}")
        y = np.loadtxt(y_path, dtype=np.int64) - 1
        return X, y

    raw_root = find_raw_root()
    X_train, y_train = load_split(raw_root, "train")
    X_test, y_test = load_split(raw_root, "test")

    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape).astype(np.float32)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    out_train = dataset_dir / "har_train.npz"
    out_test = dataset_dir / "har_test.npz"
    np.savez(out_train, X=X_train, y=y_train.astype(np.int64))
    np.savez(out_test, X=X_test, y=y_test.astype(np.int64))
    print(f"[SAVED] {out_train}")
    print(f"[SAVED] {out_test}")


def prepare_ettm1() -> None:
    dataset_dir = DATA_DIR / "ETTm1"
    csv_path = dataset_dir / "ETTm1_clean.csv"
    out_train = dataset_dir / "ett_train.npz"
    out_test = dataset_dir / "ett_test.npz"
    t_window = 128
    horizon = 4
    train_ratio = 0.8

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing ETTm1 CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ["date", "Date"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    values = df.values.astype(np.float32)

    X, y = [], []
    for t in range(len(values) - t_window - horizon):
        X.append(values[t : t + t_window])
        y.append(values[t + t_window + horizon - 1][0])
    X = np.stack(X, axis=0)
    y = np.asarray(y, dtype=np.float32)

    n_train = int(len(X) * train_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_train, X=X_train, y=y_train)
    np.savez(out_test, X=X_test, y=y_test)
    print(f"[SAVED] {out_train}")
    print(f"[SAVED] {out_test}")


def prepare_ptbxl() -> None:
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("PTB-XL preprocessing requires the optional dependency `wfdb`.") from exc

    dataset_dir = DATA_DIR / "PTBXL"
    db_csv = dataset_dir / "ptbxl_database.csv"
    scp_csv = dataset_dir / "scp_statements.csv"
    out_train = dataset_dir / "ptbxl_train.npz"
    out_test = dataset_dir / "ptbxl_test.npz"
    target_fs = 100
    t_length = 1000
    num_channels = 12
    test_fold = 10

    if not db_csv.exists() or not scp_csv.exists():
        raise FileNotFoundError(f"Missing PTB-XL metadata under {dataset_dir}")

    y_meta = pd.read_csv(db_csv, index_col="ecg_id")
    y_meta["scp_codes"] = y_meta["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(scp_csv, index_col=0)
    scp = scp[scp["diagnostic"] == 1]

    def aggregate_diagnostic(code_dict):
        labels = []
        for key in code_dict:
            if key in scp.index:
                labels.append(scp.loc[key].diagnostic_class)
        return list(set(labels))

    def to_binary(superclasses):
        if not isinstance(superclasses, (list, tuple)) or len(superclasses) == 0:
            return 0
        if len(superclasses) == 1 and "NORM" in superclasses:
            return 0
        return 1

    y_meta["diagnostic_superclass"] = y_meta["scp_codes"].apply(aggregate_diagnostic)
    y_meta["label"] = y_meta["diagnostic_superclass"].apply(to_binary)

    records = []
    for filename in y_meta["filename_lr"]:
        signal, _meta = wfdb.rdsamp(str(dataset_dir / filename))
        if signal.shape[1] != num_channels:
            raise ValueError(f"Unexpected PTB-XL channel count: {signal.shape}")
        if signal.shape[0] > t_length:
            signal = signal[:t_length]
        elif signal.shape[0] < t_length:
            signal = np.pad(signal, ((0, t_length - signal.shape[0]), (0, 0)), mode="constant")
        records.append(signal.astype(np.float32))

    X = np.stack(records, axis=0)
    y = y_meta["label"].values.astype(np.int64)
    train_idx = np.where(y_meta["strat_fold"].values != test_fold)[0]
    test_idx = np.where(y_meta["strat_fold"].values == test_fold)[0]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train.reshape(-1, num_channels))
    X_train = scaler.transform(X_train.reshape(-1, num_channels)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, num_channels)).reshape(X_test.shape)

    np.savez(out_train, X=X_train, y=y_train)
    np.savez(out_test, X=X_test, y=y_test)
    print(f"[SAVED] {out_train}")
    print(f"[SAVED] {out_test}")


def prepare_sp500() -> None:
    dataset_dir = DATA_DIR / "SP500"
    csv_path = PROJECT_ROOT / "sp500_macro_2005_2024_daily.csv"
    out_train = dataset_dir / "sp500_train.npz"
    out_test = dataset_dir / "sp500_test.npz"
    t_window = 20
    train_end = pd.to_datetime("2020-12-31")
    feature_cols = [
        "S_Open", "S_High", "S_Low", "S_Close", "S_Volume",
        "SMA_10", "SMA_20", "VIX_Close", "Gold_Close", "DXY_Close", "WTI_Close",
    ]
    label_cols = ["Return_1d", "Return_3d", "Return_7d"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing S&P500 CSV: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    close = df["S_Close"]
    df["Return_1d"] = close.pct_change().shift(-1)
    df["Return_3d"] = (close.shift(-3) - close) / close
    df["Return_7d"] = (close.shift(-7) - close) / close
    df = df.dropna()

    train_df = df.loc[df.index <= train_end].copy()
    test_df = df.loc[df.index > train_end].copy()
    scaler = StandardScaler().fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    def rolling(frame: pd.DataFrame):
        feats = frame[feature_cols].values.astype(np.float32)
        labels = frame[label_cols].values.astype(np.float32)
        dates = frame.index.to_numpy()
        X, Y, D = [], [], []
        for i in range(len(frame) - t_window):
            X.append(feats[i : i + t_window])
            Y.append(labels[i + t_window - 1])
            D.append(dates[i + t_window - 1])
        return np.stack(X), np.stack(Y), np.asarray(D)

    X_train, y_train, dates_train = rolling(train_df)
    X_test, y_test, dates_test = rolling(test_df)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_train, X=X_train, y=y_train, dates=dates_train)
    np.savez_compressed(out_test, X=X_test, y=y_test, dates=dates_test)
    print(f"[SAVED] {out_train}")
    print(f"[SAVED] {out_test}")


PREPARE = {
    "har": prepare_har,
    "ettm1": prepare_ettm1,
    "ptbxl": prepare_ptbxl,
    "sp500": prepare_sp500,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets for GS-SHAP experiments.")
    parser.add_argument("dataset", choices=sorted(PREPARE))
    args = parser.parse_args()
    PREPARE[args.dataset]()


if __name__ == "__main__":
    main()
