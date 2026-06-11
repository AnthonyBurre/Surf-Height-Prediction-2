"""Sequence neural nets (GRU / LSTM / TCN) on raw windowed channels (Phase 8.4).

Worth the cost only where temporal/phase dynamics carry signal the engineered
features can't. Each :class:`SeqForecaster` is bound to the raw observation frame
and builds context windows for whatever fold origins the harness hands it (the
engineered ``X`` is used only for its index), so it slots into the rolling-origin
backtest exactly like the other forecasters. Tune **per horizon** (longer context
for longer leads) and **average over seeds** — run-to-run variance can exceed a
hyperparameter effect.

``torch`` is the heavy/optional dependency; importing this module requires the
``forecast`` extra.
"""
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise RuntimeError(
        "forecast.neural needs PyTorch — install the 'forecast' extra "
        "(uv sync --all-extras)."
    ) from exc

from .windows import WindowScaler, windows_for_index


# --------------------------------------------------------------------------- #
# Torch modules
# --------------------------------------------------------------------------- #
class _GRU(nn.Module):
    def __init__(self, n_ch, hidden, layers, dropout, n_out):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _LSTM(nn.Module):
    def __init__(self, n_ch, hidden, layers, dropout, n_out):
        super().__init__()
        self.rnn = nn.LSTM(n_ch, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _CausalConv1d(nn.Module):
    """Left-padded dilated conv so output[t] depends only on inputs ≤ t."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, : -self.pad] if self.pad else out


class _TCN(nn.Module):
    """Compact dilated *causal* temporal conv net, channels-last input."""

    def __init__(self, n_ch, hidden, layers, dropout, n_out):
        super().__init__()
        blocks = []
        in_ch = n_ch
        for i in range(layers):
            blocks += [
                _CausalConv1d(in_ch, hidden, kernel_size=3, dilation=2 ** i),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T); take the last (most recent) timestep
        z = self.net(x.transpose(1, 2))
        return self.head(z[:, :, -1])


_ARCHS = {"gru": _GRU, "lstm": _LSTM, "tcn": _TCN}


class SeqForecaster:
    """Sklearn-style sequence forecaster bound to a raw observation frame."""

    def __init__(
        self,
        df_raw: pd.DataFrame,
        channels: Sequence[str],
        context_len: int,
        horizon_h: int,
        *,
        arch: str = "gru",
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 25,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        val_frac: float = 0.1,
        patience: int = 4,
        train_stride: int = 1,
        seed: int = 0,
        device: str | None = None,
    ):
        self.df = df_raw
        self.channels = list(channels)
        self.context_len = context_len
        self.h = horizon_h
        self.arch = arch
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.patience = patience
        self.train_stride = train_stride
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"{arch}"
        self.scaler_: WindowScaler | None = None
        self.model_: nn.Module | None = None

    def _build_windows(self, origins, y=None):
        Xw, kept = windows_for_index(self.df, self.channels, self.context_len, origins)
        if y is not None:
            yv = y.reindex(kept).to_numpy(dtype=np.float32)
            mask = ~np.isnan(yv)
            return Xw[mask], yv[mask], kept[mask]
        return Xw, kept

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeqForecaster":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        origins = X.index[:: self.train_stride] if self.train_stride > 1 else X.index
        Xw, yv, _ = self._build_windows(origins, y)
        if len(Xw) < 100:
            raise ValueError(f"{self.name}: too few clean windows ({len(Xw)}).")
        self.scaler_ = WindowScaler().fit(Xw)
        Xw = self.scaler_.transform(Xw).astype(np.float32)

        # chronological internal validation split for early stopping
        n_val = int(len(Xw) * self.val_frac)
        Xtr, ytr = Xw[:-n_val], yv[:-n_val]
        Xvl, yvl = Xw[-n_val:], yv[-n_val:]

        dev = self.device
        model = _ARCHS[self.arch](len(self.channels), self.hidden, self.layers,
                                  self.dropout, 1).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        Xtr_t = torch.from_numpy(Xtr).to(dev)
        ytr_t = torch.from_numpy(ytr).to(dev).unsqueeze(1)
        Xvl_t = torch.from_numpy(Xvl).to(dev)
        yvl_t = torch.from_numpy(yvl).to(dev).unsqueeze(1)

        best_val, best_state, bad = float("inf"), None, 0
        n = len(Xtr_t)
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch_size):
                bi = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[bi]), ytr_t[bi])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(Xvl_t), yvl_t).item()
            if v < best_val - 1e-5:
                best_val, best_state, bad = v, {k: x.detach().clone() for k, x in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        Xw, kept = self._build_windows(X.index)
        if len(Xw) == 0:
            return pd.Series(np.nan, index=X.index, name=self.name)
        Xw = self.scaler_.transform(Xw).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            t = torch.from_numpy(Xw).to(self.device)
            preds = self.model_(t).cpu().numpy().ravel()
        return pd.Series(preds, index=kept, name=self.name).reindex(X.index)


class SeedAverageForecaster:
    """Average predictions of the same SeqForecaster spec across seeds."""

    def __init__(self, make_seq, seeds: Sequence[int] = (0, 1, 2), name: str | None = None):
        self.make_seq = make_seq
        self.seeds = list(seeds)
        self.name = name or "seq_seedavg"
        self.models_: list = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeedAverageForecaster":
        self.models_ = [self.make_seq(s).fit(X, y) for s in self.seeds]
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = pd.concat([m.predict(X) for m in self.models_], axis=1)
        return preds.mean(axis=1).rename(self.name)


def context_for_horizon(horizon_h: int) -> int:
    """A sensible default context length (steps) that grows with the horizon."""
    return {6: 48, 12: 96, 24: 144, 36: 192, 48: 240, 72: 336}.get(horizon_h, 144)
