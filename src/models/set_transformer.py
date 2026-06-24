"""Set Transformer for poker chunk classification.

Key idea: treat each chunk as an UNORDERED SET of hands. Self-attention across
hands captures inter-hand relationships (consistency, drift, variance) that
stat-based aggregation misses — e.g. whether hand 3 and hand 11 use identical
bet sizes, or whether aggression monotonically drops across the session.

Architecture (lightweight for 430-example regime):
  - Linear projection: input_dim → hidden_dim
  - 2× Multihead Self-Attention blocks (SAB) with residual + LayerNorm
  - Pooling by Multihead Attention (PMA) — learns a single query vector
  - MLP head → sigmoid score

With hidden_dim=32, num_heads=4, dropout=0.4: ~4k parameters — small enough
to avoid severe overfitting on 400 training examples with cross-validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _SAB(nn.Module):
    """Set Attention Block: self-attention + residual + layernorm."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class _PMA(nn.Module):
    """Pooling by Multihead Attention: collapses set → k seed vectors."""

    def __init__(self, dim: int, num_heads: int, k: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.size(0)
        q = self.seeds.expand(b, -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return self.norm(out)  # (B, k, dim)


class SetTransformerClassifier(nn.Module):
    """Full Set Transformer: set of hand-feature vectors → bot probability."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_sab: int = 2,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.ModuleList([_SAB(hidden_dim, num_heads, dropout) for _ in range(num_sab)])
        self.pma = _PMA(hidden_dim, num_heads, k=1, dropout=dropout)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, max_hands, input_dim)
        mask: (B, max_hands) — True where padding (will be ignored by attention)
        returns: (B,) raw logits
        """
        x = self.input_proj(x)
        for sab in self.encoder:
            x = sab(x, key_padding_mask=mask)
        x = self.pma(x, key_padding_mask=mask)  # (B, 1, hidden_dim)
        return self.head(x).squeeze(-1)            # (B,)


# ── Sklearn-compatible wrapper ─────────────────────────────────────────────

class SetTransformerBot:
    """Sklearn-style wrapper: fit / predict_proba on lists of hand matrices."""

    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_sab: int = 2,
        dropout: float = 0.4,
        lr: float = 3e-3,
        epochs: int = 80,
        batch_size: int = 32,
        patience: int = 15,
        random_seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_sab = num_sab
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_seed = random_seed
        self.device = device
        self.model_: SetTransformerClassifier | None = None
        self.max_hands_: int = 0

    def _pad(self, matrices: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad variable-length hand sequences to max_hands, return (X, mask)."""
        B = len(matrices)
        max_h = max(m.shape[0] for m in matrices)
        self.max_hands_ = max(self.max_hands_, max_h)
        padded = np.zeros((B, max_h, self.input_dim), dtype=np.float32)
        mask = np.ones((B, max_h), dtype=bool)  # True = padding (ignored)
        for i, m in enumerate(matrices):
            n = min(m.shape[0], max_h)
            padded[i, :n, : m.shape[1]] = m[:n, : self.input_dim]
            mask[i, :n] = False
        return torch.tensor(padded), torch.tensor(mask)

    def fit(
        self,
        matrices_train: list[np.ndarray],
        y_train: np.ndarray,
        matrices_val: list[np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "SetTransformerBot":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.model_ = SetTransformerClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_sab=self.num_sab,
            dropout=self.dropout,
        ).to(self.device)

        x_tr, m_tr = self._pad(matrices_train)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32)
        else:
            w_t = torch.ones(len(y_t))

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        has_val = matrices_val is not None and y_val is not None
        if has_val:
            x_val, m_val = self._pad(matrices_val)
            # extend max_hands_ if val is longer
            x_tr, m_tr = self._pad(matrices_train)  # re-pad with updated max
            y_v = torch.tensor(y_val, dtype=torch.float32)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(self.epochs):
            self.model_.train()
            # Mini-batch training
            idx = torch.randperm(len(x_tr))
            epoch_loss = 0.0
            for start in range(0, len(x_tr), self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                xb = x_tr[batch_idx].to(self.device)
                mb = m_tr[batch_idx].to(self.device)
                yb = y_t[batch_idx].to(self.device)
                wb = w_t[batch_idx].to(self.device)
                logits = self.model_(xb, mb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    xv = x_val.to(self.device)
                    mv = m_val.to(self.device)
                    yv = y_v.to(self.device)
                    logits_v = self.model_(xv, mv)
                    val_loss = F.binary_cross_entropy_with_logits(logits_v, yv).item()
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, matrices: list[np.ndarray]) -> np.ndarray:
        assert self.model_ is not None, "Call fit() first"
        self.model_.eval()
        x, m = self._pad(matrices)
        with torch.no_grad():
            logits = self.model_(x.to(self.device), m.to(self.device))
            return torch.sigmoid(logits).cpu().numpy()

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "SetTransformerBot":
        import joblib
        return joblib.load(path)
