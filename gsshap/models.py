# gsshap/models.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BiLSTMSeqModel(nn.Module):
    """
    Common sequence model.
    - task="clf": classification, returns logits
    - task="reg": regression, returns scalar
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.3,
        task: str = "clf",
    ):
        super().__init__()
        assert task in ("clf", "reg")
        self.task = task

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        out = self.fc(last_hidden)
        if self.task == "reg":
            out = out.squeeze(-1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / max(1, d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerSeqModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.3,
        task: str = "clf",
        num_heads: int = 4,
        ff_mult: int = 4,
    ):
        super().__init__()
        assert task in ("clf", "reg")
        self.task = task

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = self.norm(h[:, -1, :])
        out = self.head(h)
        if self.task == "reg":
            out = out.squeeze(-1)
        return out


class _MambaBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "Mamba backbone requires the `mamba_ssm` package."
            ) from exc

        self.norm = nn.LayerNorm(hidden_dim)
        self.mixer = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mixer(self.norm(x)))


class MambaSeqModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.3,
        task: str = "clf",
    ):
        super().__init__()
        assert task in ("clf", "reg")
        self.task = task

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_MambaBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h[:, -1, :])
        out = self.head(h)
        if self.task == "reg":
            out = out.squeeze(-1)
        return out


def build_sequence_model(
    backbone: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int,
    dropout: float = 0.3,
    task: str = "clf",
    num_heads: int = 4,
) -> nn.Module:
    key = backbone.lower().strip()
    if key == "bilstm":
        return BiLSTMSeqModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            task=task,
        )
    if key == "transformer":
        return TransformerSeqModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            task=task,
            num_heads=num_heads,
        )
    if key == "mamba":
        return MambaSeqModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            task=task,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")
