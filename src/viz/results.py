"""Phase-11 headline figures. These consume the experiment log
(:func:`forecast.read_log` output) and ``EvalResult.predictions`` — no project
globals."""
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._style import DIV_CMAP, PALETTE


def skill_vs_horizon(
    df: pd.DataFrame,
    *,
    value: str = "rmse",
    models: Sequence[str] | None = None,
    baseline_name: str = "persistence",
    mode: str = "select",
    title: str | None = None,
) -> plt.Figure:
    """Fig 11.1 — the money chart: error & skill vs horizon, one line per model,
    a fold-spread band on each, the better-baseline backdrop, and stars on the
    per-horizon winner."""
    d = df[df["mode"] == mode] if (mode is not None and "mode" in df.columns) else df
    names = list(models) if models is not None else [n for n in d["name"].unique()]
    horizons = sorted(d["horizon_h"].unique())

    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    # left: error vs horizon
    for i, name in enumerate(names):
        sub = d[d["name"] == name].set_index("horizon_h").reindex(horizons)
        c = PALETTE[i % len(PALETTE)]
        ax_e.plot(horizons, sub[value], "-o", color=c, label=name, ms=4)
        if "fold_spread_rmse" in sub and value == "rmse":
            lo = sub[value] - sub["fold_spread_rmse"]
            hi = sub[value] + sub["fold_spread_rmse"]
            ax_e.fill_between(horizons, lo, hi, color=c, alpha=0.12)
    # baseline backdrop (persistence error reconstructed from skill where available)
    base = _baseline_curve(d, value, baseline_name)
    if base is not None:
        ax_e.plot(horizons, base.reindex(horizons), "--", color="0.4", lw=1.2,
                  label=f"{baseline_name} (baseline)")
    # winner stars
    best = d.loc[d.groupby("horizon_h")[value].idxmin()]
    ax_e.plot(best["horizon_h"], best[value], "*", color="gold", ms=14,
              markeredgecolor="k", zorder=5)
    ax_e.set_xlabel("horizon (h)"); ax_e.set_ylabel(f"{value.upper()} (m)")
    ax_e.set_title(f"{value.upper()} vs horizon (★ = winner; band = fold spread)")
    ax_e.legend(fontsize=8)

    # right: skill vs horizon
    for i, name in enumerate(names):
        sub = d[d["name"] == name].set_index("horizon_h").reindex(horizons)
        ax_s.plot(horizons, sub["skill_rmse"], "-o", color=PALETTE[i % len(PALETTE)],
                  label=name, ms=4)
    ax_s.axhline(0, color="0.4", ls="--", lw=1.0)
    ax_s.set_xlabel("horizon (h)"); ax_s.set_ylabel(f"skill vs {baseline_name}")
    ax_s.set_title("Skill vs horizon (>0 beats baseline)")

    fig.suptitle(title or "Error & skill vs lead time — the deployment recipe per horizon",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def _baseline_curve(d: pd.DataFrame, value: str, baseline_name: str) -> pd.Series | None:
    """Reconstruct the baseline error per horizon from a model's skill, or from a
    logged baseline row if present."""
    if baseline_name in set(d["name"]):
        b = d[d["name"] == baseline_name].set_index("horizon_h")[value]
        return b
    if value == "rmse" and "skill_rmse" in d:
        # rmse_baseline = rmse_model / (1 - skill)
        rows = d.dropna(subset=["skill_rmse"])
        if rows.empty:
            return None
        rows = rows.assign(base=rows["rmse"] / (1 - rows["skill_rmse"]))
        return rows.groupby("horizon_h")["base"].median()
    return None


def forest_plot(df: pd.DataFrame, horizon: int, *, mode: str = "select") -> plt.Figure:
    """Fig 11.2 — contender forest plot at one horizon: mean RMSE with a
    block-bootstrap CI bar; colour from the paired test vs baseline
    (sig-better / tied / worse). The winner's CI is shaded."""
    d = df[(df["horizon_h"] == horizon)]
    if mode is not None and "mode" in d.columns:
        d = d[d["mode"] == mode]
    d = d.sort_values("rmse", ascending=False).reset_index(drop=True)

    def color(row):
        if row.get("paired_sig") and (row.get("paired_delta") or 0) < 0:
            return "#009E73"   # significantly beats baseline
        if row.get("paired_sig") and (row.get("paired_delta") or 0) > 0:
            return "#D55E00"   # significantly worse
        return "0.5"           # tied with baseline

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 1.4))
    winner = d["rmse"].idxmin()
    for i, row in d.iterrows():
        lo, hi = row.get("rmse_lo", np.nan), row.get("rmse_hi", np.nan)
        ax.plot([lo, hi], [i, i], color=color(row), lw=2.5)
        ax.plot(row["rmse"], i, "o", color=color(row), ms=6)
    if np.isfinite(d.loc[winner, "rmse_lo"]):
        ax.axvspan(d.loc[winner, "rmse_lo"], d.loc[winner, "rmse_hi"],
                   color="gold", alpha=0.12)
    ax.set_yticks(range(len(d)), d["name"])
    ax.set_xlabel("RMSE (m) with bootstrap CI")
    ax.set_title(f"Contenders at +{horizon}h (green=beats baseline, grey=tie, orange=worse)")
    fig.tight_layout()
    return fig


def residual_diagnostics(predictions: pd.DataFrame, feature: pd.Series | None = None,
                         feature_name: str = "feature") -> plt.Figure:
    """Fig 11.3 — residual vs prediction hexbin with marginals, plus residual vs a
    key feature with quantile bands. Reads heteroscedasticity & tail bias."""
    p = predictions.dropna(subset=["y_true", "y_pred"]).copy()
    p["resid"] = p["y_true"] - p["y_pred"]

    ncol = 2 if feature is not None else 1
    fig = plt.figure(figsize=(6.5 * ncol, 4.6))
    gs = fig.add_gridspec(2, 2 * ncol, height_ratios=[1, 4], width_ratios=[4, 1] * ncol,
                          hspace=0.05, wspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax)
    axright = fig.add_subplot(gs[1, 1], sharey=ax)
    hb = ax.hexbin(p["y_pred"], p["resid"], gridsize=45, cmap="viridis", mincnt=1, bins="log")
    ax.axhline(0, color="white", lw=1.0)
    ax.set_xlabel("prediction (m)"); ax.set_ylabel("residual (true − pred)")
    axtop.hist(p["y_pred"], bins=60, color="0.6"); axtop.axis("off")
    axtop.set_title("Residual vs prediction")
    axright.hist(p["resid"], bins=60, orientation="horizontal", color="0.6"); axright.axis("off")
    fig.colorbar(hb, ax=axright, fraction=0.2, pad=0.02, label="log count")

    if feature is not None:
        axf = fig.add_subplot(gs[1, 2])
        f = feature.reindex(p.index)
        bins = pd.qcut(f, 12, duplicates="drop")
        g = p.assign(f=f).groupby(bins, observed=True)["resid"]
        centers = [iv.mid for iv in g.median().index]
        axf.plot(centers, g.median().to_numpy(), "-o", color="#0072B2", ms=3, label="median")
        axf.fill_between(centers, g.quantile(0.1).to_numpy(), g.quantile(0.9).to_numpy(),
                         color="#0072B2", alpha=0.15, label="P10–P90")
        axf.axhline(0, color="0.4", ls="--")
        axf.set_xlabel(feature_name); axf.set_ylabel("residual")
        axf.set_title(f"Residual vs {feature_name}")
        axf.legend(fontsize=8)

    fig.suptitle("Residual diagnostics (tail bias & heteroscedasticity)",
                 y=1.02, fontsize=11, fontweight="bold")
    return fig


def importance_horizon(importance: pd.DataFrame, drop_one: pd.DataFrame | None = None,
                       titles=("add-one gain", "drop-one cost")) -> plt.Figure:
    """Fig 11.4 — input × horizon effect heatmap. With ``drop_one`` provided,
    renders two panels (grouped-source add-one vs drop-one ablation)."""
    panels = [(importance, titles[0])]
    if drop_one is not None:
        panels.append((drop_one, titles[1]))
    vmax = max(np.nanmax(np.abs(p.to_numpy())) for p, _ in panels) or 1.0

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 0.4 * len(importance) + 1.6),
                             squeeze=False)
    for ax, (P, title) in zip(axes[0], panels):
        im = ax.imshow(P.to_numpy(), aspect="auto", cmap=DIV_CMAP, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(P.shape[1]), [f"{c}h" for c in P.columns])
        ax.set_yticks(range(P.shape[0]), P.index)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Importance / source ablation (input × horizon)", y=1.02,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
