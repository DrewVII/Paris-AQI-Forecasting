"""
Direct multi-horizon AQI forecaster with quantile outputs.

A GRU encodes the lookback window; a small head emits every horizon and every
quantile at once. Output shape is (B, H, Q).

Quantiles are built to be monotone: the head predicts the lowest quantile
directly and each higher one as a strictly positive increment on top of the
last. Without this, independently trained quantiles can cross (a "90th
percentile" below the median), which is nonsense and happens more often than
you'd expect on small data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AQIForecaster(nn.Module):
    def __init__(self, n_features, horizon=24, hidden=64, layers=2,
                 dropout=0.2, n_quantiles=1):
        super().__init__()
        self.H = horizon
        self.Q = n_quantiles
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon * n_quantiles),
        )

    def pool(self, out):
        """Summarise the encoded window. Base model: last hidden state."""
        return out[:, -1]

    def forward(self, x):
        # x: (B, L, F)
        out, _ = self.gru(x)
        raw = self.head(self.pool(out)).view(-1, self.H, self.Q)
        if self.Q == 1:
            return raw
        base = raw[..., :1]
        steps = F.softplus(raw[..., 1:])  # > 0, so quantiles can't cross
        return torch.cat([base, base + torch.cumsum(steps, dim=-1)], dim=-1)


class AQIForecasterAttn(AQIForecaster):
    """Same encoder with additive attention pooling over the window instead
    of using only the final hidden state. Marginally better when the useful
    signal sits mid-window (e.g. yesterday afternoon's ozone peak)."""

    def __init__(self, n_features, horizon=24, hidden=64, layers=2,
                 dropout=0.2, n_quantiles=1):
        super().__init__(n_features, horizon, hidden, layers, dropout,
                         n_quantiles)
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

    def pool(self, out):
        w = torch.softmax(self.attn(out), dim=1)  # (B, L, 1)
        return (out * w).sum(dim=1)