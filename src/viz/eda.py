"""Phase-2 EDA figures. Each is multi-dimensional and ends in a modelling
decision. All functions take plain pandas objects and return a Matplotlib
``Figure`` (saved by the caller via :func:`viz.save`)."""
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._style import DIV_CMAP, SEQ_CMAP


def coverage_matrix(frame: pd.DataFrame, freq: str = "MS") -> plt.Figure:
    """Fig 2.1 — availability matrix: x=time bin, y=series, colour=fraction present.

    Rows ordered by first-valid timestamp. Reads the (source × time ×
    completeness) trade space; drives the length-vs-breadth call and the
    sparse-column drop threshold.
    """
    present = frame.notna().resample(freq).mean()
    # order rows by first time a column is >50% present
    first_valid = {c: (present[c] > 0.5).idxmax() if (present[c] > 0.5).any()
                   else present.index[-1] for c in present.columns}
    order = sorted(present.columns, key=lambda c: first_valid[c])
    M = present[order].to_numpy().T

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(order) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)), order)
    n = len(present.index)
    ticks = np.linspace(0, n - 1, min(n, 10)).astype(int)
    ax.set_xticks(ticks, [present.index[t].strftime("%Y-%m") for t in ticks], rotation=45, ha="right")
    ax.set_title(f"Coverage / availability ({freq} bins) — fraction of non-missing readings")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="fraction present")
    fig.tight_layout()
    return fig


def target_distribution(y: pd.Series) -> plt.Figure:
    """Fig 2.2 — target distribution raw vs log vs Yeo-Johnson, with an ECDF inset."""
    from sklearn.preprocessing import PowerTransformer

    v = y.dropna().to_numpy()
    logv = np.log(v + 1e-3)
    yj = PowerTransformer(method="yeo-johnson").fit_transform(v.reshape(-1, 1)).ravel()
    from scipy.stats import skew

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, data, title in zip(
        axes, (v, logv, yj), ("raw hsig_m", "log(hsig_m)", "Yeo-Johnson")
    ):
        ax.hist(data, bins=80, color="#0072B2", alpha=0.8)
        ax.set_title(f"{title}\nskew={skew(data):.2f}")
        ax.set_xlabel(title)
    axes[0].set_ylabel("count")
    # ECDF inset on the raw panel
    ins = axes[0].inset_axes([0.55, 0.5, 0.42, 0.42])
    xs = np.sort(v)
    ins.plot(xs, np.linspace(0, 1, len(xs)), color="#D55E00", lw=1.2)
    ins.set_title("ECDF", fontsize=7)
    ins.tick_params(labelsize=6)
    fig.suptitle("Target distribution — raw, log, Yeo-Johnson (drives transform choice)",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def decomposition(y: pd.Series, period: int = 365) -> plt.Figure:
    """Fig 2.3 — STL on the daily-mean series + rolling mean/var + ADF/KPSS +
    a periodogram of the native-cadence series (to discover cycle lengths)."""
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss
    from scipy.signal import periodogram

    daily = y.resample("1D").mean().interpolate("time").dropna()
    stl = STL(daily, period=period, robust=True).fit()

    fig, axes = plt.subplots(3, 2, figsize=(12, 7.5))
    axes[0, 0].plot(daily.index, daily.to_numpy(), lw=0.6, color="#0072B2")
    axes[0, 0].set_title("Observed (daily mean)")
    axes[1, 0].plot(daily.index, stl.trend.to_numpy(), lw=0.9, color="#000000")
    axes[1, 0].set_title("STL trend")
    axes[2, 0].plot(daily.index, stl.seasonal.to_numpy(), lw=0.5, color="#009E73")
    axes[2, 0].set_title(f"STL seasonal (period={period} d)")

    # rolling mean / variance strip
    roll = daily.rolling(30)
    axes[0, 1].plot(daily.index, roll.mean().to_numpy(), color="#0072B2", label="mean")
    axes[0, 1].plot(daily.index, roll.std().to_numpy(), color="#D55E00", label="std")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_title("30-day rolling mean / std (level & variance drift)")

    # residual
    axes[1, 1].plot(daily.index, stl.resid.to_numpy(), lw=0.4, color="#888888")
    axes[1, 1].set_title("STL remainder")

    # periodogram of native cadence (interpolated)
    fine = y.interpolate("time").dropna().to_numpy()
    f, P = periodogram(fine - fine.mean(), fs=48.0)  # cycles per day
    axes[2, 1].semilogy(f, P, lw=0.6, color="#CC79A7")
    for c, lab in [(1, "1/day"), (1 / 365.0, "1/yr")]:
        axes[2, 1].axvline(c, color="k", ls=":", lw=0.8)
    axes[2, 1].set_xlim(0, 4)
    axes[2, 1].set_title("Periodogram (cycles/day)")
    axes[2, 1].set_xlabel("frequency (cycles/day)")

    adf_p = adfuller(daily.to_numpy())[1]
    try:
        kpss_p = kpss(daily.to_numpy(), regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = float("nan")
    fig.suptitle(
        f"Decomposition & stationarity — ADF p={adf_p:.3f} (small ⇒ stationary), "
        f"KPSS p={kpss_p:.3f} (small ⇒ non-stationary)",
        y=1.02, fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def acf_pacf(frames: Mapping[str, pd.Series], max_lag: int = 192) -> plt.Figure:
    """Fig 2.4 — ACF/PACF vs lag, multiple series overlaid, plus the persistence
    error-vs-horizon implied by the target ACF (sqrt(2(1-rho)) * sd)."""
    from statsmodels.tsa.stattools import acf, pacf

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    target_name = next(iter(frames))
    lags_h = np.arange(max_lag + 1) / 2.0  # 30-min steps -> hours
    for name, s in frames.items():
        v = s.dropna().to_numpy()
        a = acf(v, nlags=max_lag, fft=True)
        axes[0].plot(lags_h, a, lw=1.0, label=name)
        p = pacf(v, nlags=min(max_lag, 100))
        axes[1].plot(np.arange(len(p)) / 2.0, p, lw=1.0, label=name)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("ACF"); axes[0].set_xlabel("lag (hours)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("PACF"); axes[1].set_xlabel("lag (hours)")
    axes[0].legend();

    tgt = frames[target_name].dropna().to_numpy()
    a = acf(tgt, nlags=max_lag, fft=True)
    sd = tgt.std()
    axes[2].plot(lags_h, sd * np.sqrt(np.clip(2 * (1 - a), 0, None)), color="#D55E00", lw=1.3)
    axes[2].set_title("Implied persistence RMSE vs lead time")
    axes[2].set_xlabel("lead time (hours)"); axes[2].set_ylabel("RMSE (m)")
    fig.suptitle("Autocorrelation across lead time (places the lag grid)",
                 y=1.04, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def seasonality_calendar(y: pd.Series) -> plt.Figure:
    """Fig 2.5 — month × hour-of-day heatmaps of mean and variance, with marginals."""
    df = pd.DataFrame({"y": y.dropna()})
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    mean_t = df.pivot_table("y", "hour", "month", aggfunc="mean")
    var_t = df.pivot_table("y", "hour", "month", aggfunc="var")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, T, title, cmap in (
        (axes[0], mean_t, "mean hsig_m", SEQ_CMAP),
        (axes[1], var_t, "variance hsig_m", "magma"),
    ):
        im = ax.imshow(T.to_numpy(), aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(12), [f"{m}" for m in T.columns])
        ax.set_yticks(range(0, 24, 3), [f"{h}" for h in range(0, 24, 3)])
        ax.set_xlabel("month"); ax.set_ylabel("hour of day")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Seasonality calendar — annual × diurnal structure (drives calendar features)",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def lead_lag_matrix(frame: pd.DataFrame, target_col: str, max_lag: int = 48,
                    columns: Sequence[str] | None = None) -> plt.Figure:
    """Fig 2.6 — for each companion, the peak cross-correlation with the target and
    the lag at which it occurs (positive lag = companion leads the target)."""
    cols = list(columns) if columns is not None else [c for c in frame.columns if c != target_col]
    tgt = frame[target_col]
    lags = range(-max_lag, max_lag + 1)
    best_corr, best_lag = [], []
    for c in cols:
        s = frame[c]
        corrs = [tgt.corr(s.shift(L)) for L in lags]
        corrs = np.array(corrs, dtype=float)
        j = int(np.nanargmax(np.abs(corrs)))
        best_corr.append(corrs[j]); best_lag.append(list(lags)[j] / 2.0)  # hours

    order = np.argsort(best_corr)[::-1]
    cols = [cols[i] for i in order]
    best_corr = [best_corr[i] for i in order]
    best_lag = [best_lag[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(cols) + 1.5))
    colors = ["#0072B2" if c >= 0 else "#D55E00" for c in best_corr]
    ax.barh(range(len(cols)), best_corr, color=colors)
    for i, (c, lag) in enumerate(zip(best_corr, best_lag)):
        ax.text(c, i, f"  lag {lag:+.1f}h", va="center",
                ha="left" if c >= 0 else "right", fontsize=7)
    ax.set_yticks(range(len(cols)), cols)
    ax.invert_yaxis()
    ax.set_xlabel(f"peak |corr| with {target_col}")
    ax.set_title("Cross-source lead–lag (annotation = lag at peak; +lag ⇒ companion leads)")
    fig.tight_layout()
    return fig


def feature_horizon_screen(X: pd.DataFrame, y_by_h: Mapping[int, pd.Series],
                           method: str = "mutual_info", top: int = 25) -> plt.Figure:
    """Fig 2.7 — feature × horizon predictive-strength heatmap (|corr| or MI)."""
    from sklearn.feature_selection import mutual_info_regression

    horizons = list(y_by_h)
    scores = {}
    for h in horizons:
        y = y_by_h[h]
        common = X.index.intersection(y.dropna().index)
        Xc = X.loc[common].fillna(X.median(numeric_only=True))
        yc = y.loc[common]
        # subsample for speed
        if len(common) > 20000:
            sub = np.random.default_rng(0).choice(len(common), 20000, replace=False)
            Xc, yc = Xc.iloc[sub], yc.iloc[sub]
        if method == "mutual_info":
            s = mutual_info_regression(Xc.to_numpy(), yc.to_numpy(), random_state=0)
        else:
            s = np.abs([np.corrcoef(Xc[c], yc)[0, 1] for c in Xc.columns])
        scores[h] = pd.Series(s, index=X.columns)
    S = pd.DataFrame(scores)
    keep = S.max(axis=1).sort_values(ascending=False).head(top).index
    S = S.loc[keep]

    fig, ax = plt.subplots(figsize=(1.1 * len(horizons) + 3, 0.32 * len(keep) + 1.5))
    im = ax.imshow(S.to_numpy(), aspect="auto", cmap=SEQ_CMAP)
    ax.set_xticks(range(len(horizons)), [f"{h}h" for h in horizons])
    ax.set_yticks(range(len(keep)), keep)
    ax.set_title(f"Feature × horizon predictive screen ({method})")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=method)
    fig.tight_layout()
    return fig
