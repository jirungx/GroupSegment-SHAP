# gsshap/training.py
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import BiLSTMSeqModel, build_sequence_model


def train_bilstm_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_pos: float = 1.0,
    device: torch.device = None,
    model_path: str = None,
    log_path: str = None,
) -> Tuple[torch.nn.Module, float, float]:
    """Train a BiLSTM classifier and return the best validation checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMSeqModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=num_classes,
        dropout=0.2,
        task="clf",
    ).to(device)

    if num_classes == 2 and weight_pos != 1.0:
        # Use class weighting for binary imbalance when requested.
        class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Track validation accuracy for checkpoint selection.
        model.eval()
        correct_val = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                pred = logits.argmax(dim=1)
                correct_val += (pred == y).sum().item()
        val_acc = correct_val / len(val_loader.dataset)

        print(f"[Base] Epoch {epoch+1}/{num_epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Base] Best val_acc = {best_val_acc:.4f}")

    # Report test accuracy using the selected checkpoint.
    model.eval()
    correct_test = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct_test += (pred == y).sum().item()
    test_acc = correct_test / len(test_loader.dataset)
    print(f"[Base] Test accuracy = {test_acc:.4f}")

    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print("Model saved to:", model_path)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"best_val_acc={best_val_acc:.4f}\n")
            f.write(f"test_acc={test_acc:.4f}\n")

    return model, best_val_acc, test_acc


def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return float(correct / max(total, 1))


def evaluate_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    mse_loss = torch.nn.MSELoss(reduction="sum")

    total_loss = 0.0
    total_count = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()
            if y_batch.ndim == 2 and y_batch.shape[-1] == 1:
                y_batch = y_batch.squeeze(-1)

            out = model(x_batch)
            if out.ndim == 2 and out.shape[-1] == 1:
                out = out.squeeze(-1)

            loss = mse_loss(out, y_batch)
            total_loss += loss.item()
            total_count += y_batch.numel()

            preds_list.append(out.detach().cpu().numpy())
            targets_list.append(y_batch.detach().cpu().numpy())

    if total_count == 0:
        return float("nan"), float("nan")

    mse = float(total_loss / total_count)
    preds_all = np.concatenate(preds_list, axis=0).reshape(-1)
    targets_all = np.concatenate(targets_list, axis=0).reshape(-1)
    rmse = float(np.sqrt(np.mean((targets_all - preds_all) ** 2)))
    return mse, rmse


def train_sequence_classifier(
    backbone: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_pos: float = 1.0,
    device: Optional[torch.device] = None,
    model_path: Optional[str] = None,
    log_path: Optional[str] = None,
    dropout: float = 0.2,
    num_heads: int = 4,
) -> Tuple[torch.nn.Module, float, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_sequence_model(
        backbone=backbone,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=num_classes,
        dropout=dropout,
        task="clf",
        num_heads=num_heads,
    ).to(device)

    if num_classes == 2 and weight_pos != 1.0:
        class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_count += y.numel()

        train_loss = total_loss / max(total_count, 1)
        val_acc = evaluate_classifier(model, val_loader, device=device)
        history.append((epoch + 1, train_loss, val_acc))
        print(
            f"[{backbone}] Epoch {epoch+1}/{num_epochs} "
            f"loss={train_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate_classifier(model, test_loader, device=device)
    print(f"[{backbone}] Best val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}")

    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_acc\n")
            for epoch, train_loss, val_acc in history:
                f.write(f"{epoch},{train_loss:.6f},{val_acc:.6f}\n")
            f.write(f"best_val_acc={best_val_acc:.6f}\n")
            f.write(f"test_acc={test_acc:.6f}\n")

    return model, best_val_acc, test_acc


def train_sequence_regressor(
    backbone: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    model_path: Optional[str] = None,
    log_path: Optional[str] = None,
    dropout: float = 0.2,
    num_heads: int = 4,
) -> Tuple[torch.nn.Module, float, float]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_sequence_model(
        backbone=backbone,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        task="reg",
        num_heads=num_heads,
    ).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_rmse = float("inf")
    best_state = None
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float()
            if y_batch.ndim == 2 and y_batch.shape[-1] == 1:
                y_batch = y_batch.squeeze(-1)

            optimizer.zero_grad()
            out = model(x_batch)
            if out.ndim == 2 and out.shape[-1] == 1:
                out = out.squeeze(-1)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.numel()
            running_loss += loss.item() * batch_size
            running_count += batch_size

        train_mse = running_loss / max(running_count, 1)
        val_mse, val_rmse = evaluate_regression(model, val_loader, device=device)
        history.append((epoch + 1, train_mse, val_mse, val_rmse))
        print(
            f"[{backbone}] Epoch {epoch+1}/{num_epochs} "
            f"train_mse={train_mse:.4f} val_mse={val_mse:.4f} val_rmse={val_rmse:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_rmse = evaluate_regression(model, test_loader, device=device)
    print(f"[{backbone}] Best val_rmse={best_val_rmse:.4f} test_rmse={test_rmse:.4f}")

    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_mse,val_mse,val_rmse\n")
            for epoch, train_mse, val_mse, val_rmse in history:
                f.write(f"{epoch},{train_mse:.6f},{val_mse:.6f},{val_rmse:.6f}\n")
            f.write(f"best_val_rmse={best_val_rmse:.6f}\n")
            f.write(f"test_mse={test_mse:.6f}\n")
            f.write(f"test_rmse={test_rmse:.6f}\n")

    return model, best_val_rmse, test_rmse
