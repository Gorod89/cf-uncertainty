"""Shared network parts: normaliser, dropout with explicit masks, LSTM regressor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cf_unc.constants import N_FEATURES


class Normalizer(nn.Module):
    """Affine standardisation of inputs (per feature) and of the target."""

    def __init__(self, n_features: int = N_FEATURES) -> None:
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(n_features))
        self.register_buffer("x_std", torch.ones(n_features))
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))

    @torch.no_grad()
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Normalizer":
        """X (N, H, F) or (N, F); y any shape (all entries are pooled)."""
        flat = np.asarray(X, dtype=np.float64).reshape(-1, X.shape[-1])
        self.x_mean.copy_(torch.as_tensor(flat.mean(axis=0), dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(np.maximum(flat.std(axis=0), 1e-6), dtype=torch.float32))
        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            self.y_mean.fill_(float(y.mean()))
            self.y_std.fill_(max(float(y.std()), 1e-6))
        return self

    def norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def norm_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def denorm_mu(self, mu_n: torch.Tensor) -> torch.Tensor:
        return self.y_mean + self.y_std * mu_n

    def denorm_sigma(self, sigma_n: torch.Tensor) -> torch.Tensor:
        return self.y_std * sigma_n


class MaskedDropout(nn.Module):
    """Dropout that can stay active at inference and can reuse explicit masks.

    An explicit ``mask`` (already scaled by 1/(1-p)) takes precedence; otherwise
    units are dropped when the module is in training mode or ``force`` is set.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = float(p)
        self.force = False

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            return x * mask
        if self.p > 0.0 and (self.training or self.force):
            return F.dropout(x, self.p, training=True)
        return x

    def sample_mask(self, n: int, width: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
        keep = torch.rand((n, width), generator=generator, device=device) >= self.p
        return keep.to(torch.float32) / (1.0 - self.p)


class SeqRegressor(nn.Module):
    """LSTM over the normalised history, last state -> one hidden layer -> ``out_dim`` raw outputs.

    Dropout sits after the LSTM state and after the hidden layer (design.md, MCDropout).
    """

    def __init__(
        self,
        out_dim: int,
        n_features: int = N_FEATURES,
        hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.head_hidden = head_hidden
        self.encoder = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True)
        self.drop1 = MaskedDropout(dropout)
        self.fc1 = nn.Linear(hidden, head_hidden)
        self.drop2 = MaskedDropout(dropout)
        self.fc2 = nn.Linear(head_hidden, out_dim)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x_norm)
        return out[:, -1]

    def forward(self, x_norm: torch.Tensor, masks: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        h = self.drop1(self.encode(x_norm), None if masks is None else masks[0])
        h = F.silu(self.fc1(h))
        h = self.drop2(h, None if masks is None else masks[1])
        return self.fc2(h)

    def force_dropout(self, flag: bool) -> None:
        self.drop1.force = flag
        self.drop2.force = flag

    def sample_masks(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.fc1.weight.device
        return (
            self.drop1.sample_mask(n, self.hidden, generator, device),
            self.drop2.sample_mask(n, self.head_hidden, generator, device),
        )
