"""Cross-source correlation of hsig_m across Mooloolaba and three southern
neighbours (Caloundra, Brisbane, Gold Coast).

Run:  ./.venv/bin/python notebooks/buoy_correlation.py

Saves three PNGs to notebooks/figures/:
  1. cross_source_corr_zero_lag.png   — Pearson r matrix at zero lag
  2. lagged_cross_corr.png            — corr vs lag, ±12h
  3. neighbour_predictive.png         — neighbour(t) vs Mooloolaba(t+h)

Question we're answering: do any of the southern neighbours carry useful
*lead* information for forecasting Mooloolaba's hsig_m at +12h? If a neighbour
is genuinely upstream of the dominant swell direction, its observations at
time t should correlate more strongly with Mooloolaba at t+h than with
Mooloolaba at t — that's a real predictive signal beyond persistence.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_buoy(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / filename,
        parse_dates=["datetime_utc"],
        index_col="datetime_utc",
    )


def main() -> None:
    buoys = {
        "mooloolaba": load_buoy("mooloolaba_wave_data_2015-2025.csv"),
        "caloundra":  load_buoy("caloundra_wave_data_2024-2025.csv"),
        "brisbane":   load_buoy("brisbane_wave_data_2024-2025.csv"),
        "gold-coast": load_buoy("gold-coast_wave_data_2024-2025.csv"),
    }

    # Mooloolaba spans 2015-2025, neighbours only 2024-2025. Restrict to overlap
    # so correlation is computed on the same time window for every pair.
    start = max(df.index.min() for df in buoys.values())
    end   = min(df.index.max() for df in buoys.values())
    print(f"Overlap window: {start} → {end}")
    for name in buoys:
        buoys[name] = buoys[name].loc[start:end]
        s = buoys[name]["hsig_m"]
        print(f"  {name:11s} {len(s):>7,} rows   hsig_m NaN: {s.isna().mean()*100:5.1f}%")

    sources = {name: df["hsig_m"] for name, df in buoys.items()}
    mool = buoys["mooloolaba"]["hsig_m"]

    # --- Plot 1: zero-lag cross-source correlation matrix -----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax, corr = viz.cross_source_correlation(sources, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cross_source_corr_zero_lag.png", dpi=150)
    plt.close(fig)
    print("\n=== Zero-lag Pearson r (hsig_m) ===")
    print(corr.round(3).to_string())

    # --- Plot 2: lagged cross-correlation curves --------------------------
    # Positive lag h: neighbour shifted forward by h hours (i.e. neighbour at
    # t-h compared to Mooloolaba at t) — measures whether the neighbour's
    # past predicts Mooloolaba's present.
    lags_h = np.arange(-12, 13, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    print("\n=== Peak correlation lag (Mooloolaba ~ neighbour at t-lag) ===")
    for name in ("caloundra", "brisbane", "gold-coast"):
        s = buoys[name]["hsig_m"]
        rs = [s.shift(int(round(h * 2))).corr(mool) for h in lags_h]
        ax.plot(lags_h, rs, marker="o", label=name)
        peak = lags_h[int(np.nanargmax(rs))]
        print(f"  {name:11s} peak r={max(rs):.3f} at lag={peak:+d}h")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set(xlabel="Lag (hours) - positive lag means neighbour leads Mooloolaba",
           ylabel="Pearson r (vs Mooloolaba hsig_m)",
           title="Lagged cross-correlation of hsig_m")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lagged_cross_corr.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: neighbour-as-feature × Mooloolaba-horizon heatmap --------
    # Bottom-line predictive question: corr(neighbour@t, Mooloolaba@t+h).
    # If column "+12h" has a higher r than "+0h" for any neighbour, that
    # neighbour carries genuine forecast lead.
    horizons_h = (0, 1, 3, 6, 12, 24, 48)
    df_n = pd.DataFrame(
        {n: buoys[n]["hsig_m"] for n in ("caloundra", "brisbane", "gold-coast")}
    )
    df_n["mooloolaba"] = mool
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax, grid = viz.feature_horizon_heatmap(
        df_n,
        target_col="mooloolaba",
        feature_cols=["caloundra", "brisbane", "gold-coast"],
        horizons_h=horizons_h,
        sampling_freq_min=30,
        ax=ax,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "neighbour_predictive.png", dpi=150)
    plt.close(fig)
    print("\n=== Pearson r (neighbour at t  vs  Mooloolaba at t+h) ===")
    print(grid.round(3).to_string())

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
