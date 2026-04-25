"""Multi-buoy EDA: coverage, distributions, seasonality, cross-source correlation.

Run:  ./.venv/bin/python notebooks/buoy_eda.py

Saves five PNGs to notebooks/figures/:
  buoy_coverage.png             — % valid hsig_m per buoy * year
  buoy_distributions.png        — violin distributions + annual trend lines
  buoy_seasonality.png          — monthly climatology + peak direction roses
  cross_source_full.png         — zero-lag Pearson r matrix + lagged ±24 h curves
  neighbour_predictive_full.png — corr(neighbour@t, Mooloolaba@t+h) heatmap
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import viz

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

BUOY_FILES: dict[str, str] = {
    "mooloolaba":        "mooloolaba_wave_data_2015-2025.csv",
    "caloundra":         "caloundra_wave_data_2013-2025.csv",
    "brisbane":          "brisbane_wave_data_2015-2025.csv",
    "gold-coast":        "gold-coast_wave_data_2015-2025.csv",
    "north-moreton-bay": "north-moreton-bay_wave_data_2010-2025.csv",
}

COLORS: dict[str, str] = {
    "mooloolaba":        "#1f77b4",
    "caloundra":         "#ff7f0e",
    "brisbane":          "#2ca02c",
    "gold-coast":        "#d62728",
    "north-moreton-bay": "#9467bd",
}

NEIGHBOURS = ["caloundra", "brisbane", "gold-coast", "north-moreton-bay"]


def load_all() -> dict[str, pd.DataFrame]:
    buoys: dict[str, pd.DataFrame] = {}
    print("Buoy summary (full history):")
    for name, fname in BUOY_FILES.items():
        df = pd.read_csv(
            DATA_DIR / fname,
            parse_dates=["datetime_utc"],
            index_col="datetime_utc",
        )
        buoys[name] = df
        hs = df["hsig_m"]
        print(
            f"  {name:20s}  {df.index.year.min()}-{df.index.year.max()}"
            f"  rows={len(df):>8,}  NaN={hs.isna().mean()*100:5.1f}%"
            f"  mean={hs.mean():.2f} m  p95={hs.quantile(0.95):.2f} m"
            f"  max={hs.max():.2f} m"
        )
    return buoys


# --------------------------------------------------------------------------- #
# Figure 1 — Data coverage heatmap
# --------------------------------------------------------------------------- #
def plot_coverage(buoys: dict[str, pd.DataFrame]) -> None:
    years = list(range(2010, 2026))
    rows = []
    for name, df in buoys.items():
        row: dict[int, float] = {}
        for yr in years:
            mask = df.index.year == yr
            total = int(mask.sum())
            if total == 0:
                row[yr] = np.nan
            else:
                row[yr] = float(df.loc[mask, "hsig_m"].notna().sum()) / total * 100
        rows.append(pd.Series(row, name=name))
    coverage = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_facecolor("#cccccc")  # gray = not deployed
    sns.heatmap(
        coverage,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=70,
        vmax=100,
        mask=coverage.isna(),
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "% valid hsig_m", "shrink": 0.8},
        ax=ax,
    )
    ax.set(
        title="Data completeness — % valid hsig_m rows per buoy per year  (grey = not deployed)",
        xlabel="Year",
        ylabel="",
    )
    plt.tight_layout()
    out = FIG_DIR / "buoy_coverage.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")

    # Text summary: flag any year with < 80 % valid
    print("  Years with < 80 % valid hsig_m:")
    for name in coverage.index:
        for yr in years:
            v = coverage.loc[name, yr]
            if not np.isnan(v) and v < 80:
                print(f"    {name} {yr}: {v:.0f}%")


# --------------------------------------------------------------------------- #
# Figure 2 — Distributions (violin) + annual mean trend
# --------------------------------------------------------------------------- #
def plot_distributions_and_trends(buoys: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel — violin per buoy across full history
    ax = axes[0]
    names = list(BUOY_FILES)
    data = [buoys[n]["hsig_m"].dropna().values for n in names]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(list(COLORS.values())[i])
        body.set_alpha(0.72)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.2)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.replace("-", "-\n") for n in names], fontsize=8)
    ax.set(ylabel="hsig_m (m)", title="Wave height distributions — full history")
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    ax.grid(axis="y", alpha=0.3)

    # Right panel — annual mean per buoy, years with > 50 % valid data only
    ax = axes[1]
    for name, df in buoys.items():
        hs = df["hsig_m"]
        annual_mean  = hs.groupby(df.index.year).mean()
        annual_valid = hs.groupby(df.index.year).apply(lambda s: s.notna().mean())
        annual_mean  = annual_mean[annual_valid > 0.5]
        ax.plot(
            annual_mean.index, annual_mean.values,
            marker="o", linewidth=1.5, label=name, color=COLORS[name],
        )
    ax.set(
        xlabel="Year",
        ylabel="Annual mean hsig_m (m)",
        title="Inter-annual variability (years > 50 % valid)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "buoy_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 3 — Seasonal climatology + peak direction roses
# --------------------------------------------------------------------------- #
def plot_seasonality_and_direction(buoys: dict[str, pd.DataFrame]) -> None:
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = plt.figure(figsize=(14, 5))
    ax_season = fig.add_subplot(1, 2, 1)
    ax_dir    = fig.add_subplot(1, 2, 2, projection="polar")

    for name, df in buoys.items():
        monthly_median = df["hsig_m"].groupby(df.index.month).median()
        ax_season.plot(
            monthly_median.index, monthly_median.values,
            marker="o", linewidth=1.5, label=name, color=COLORS[name],
        )
    ax_season.set_xticks(range(1, 13))
    ax_season.set_xticklabels(month_labels, fontsize=8)
    ax_season.set(ylabel="Median hsig_m (m)", title="Seasonal climatology (all years pooled)")
    ax_season.legend(fontsize=8)
    ax_season.grid(alpha=0.3)

    # Polar direction rose — 10° bins
    bin_edges = np.linspace(0, 2 * np.pi, 37)
    bin_width = bin_edges[1] - bin_edges[0]
    for name, df in buoys.items():
        dirs_rad = np.deg2rad(df["peak_dir_deg"].dropna().values)
        counts, _ = np.histogram(dirs_rad, bins=bin_edges)
        fracs = counts / counts.sum()
        ax_dir.bar(bin_edges[:-1], fracs, width=bin_width, alpha=0.45,
                   label=name, color=COLORS[name])
    ax_dir.set_theta_zero_location("N")
    ax_dir.set_theta_direction(-1)
    ax_dir.set_title("Peak swell direction distribution", pad=14)
    ax_dir.legend(loc="lower left", bbox_to_anchor=(-0.18, -0.12), fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "buoy_seasonality.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")

    # Print seasonal peak months
    print("  Seasonal peak (month of highest median hsig_m):")
    for name, df in buoys.items():
        monthly = df["hsig_m"].groupby(df.index.month).median()
        peak_m  = int(monthly.idxmax())
        print(f"    {name:20s}  peak month = {month_labels[peak_m - 1]}"
              f"  ({monthly.max():.2f} m median)")


# --------------------------------------------------------------------------- #
# Figure 4 — Cross-source: zero-lag matrix + lagged ±24 h curves
# --------------------------------------------------------------------------- #
def plot_cross_source(buoys: dict[str, pd.DataFrame]) -> None:
    start = max(df.index.min() for df in buoys.values())
    end   = min(df.index.max() for df in buoys.values())
    print(f"\n  Overlap window: {start.date()} → {end.date()}")

    sources = {name: buoys[name].loc[start:end, "hsig_m"] for name in BUOY_FILES}
    mool = sources["mooloolaba"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax, corr = viz.cross_source_correlation(sources, ax=axes[0])
    print("\n  Zero-lag Pearson r (hsig_m, overlap window):")
    print(corr.round(3).to_string())

    # Lagged cross-correlation ±24 h at 1-hour steps
    lags_h  = np.arange(-24, 25, 1)
    ax = axes[1]
    print("\n  Peak lagged correlation (neighbour leads Mooloolaba):")
    for name in NEIGHBOURS:
        s  = sources[name]
        rs = [s.shift(int(round(h * 2))).corr(mool) for h in lags_h]
        ax.plot(lags_h, rs, linewidth=1.5, label=name, color=COLORS[name])
        peak_lag = lags_h[int(np.nanargmax(rs))]
        print(f"    {name:20s}  peak r={max(rs):.3f} at lag={peak_lag:+d} h")

    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set(
        xlabel="Lag h  (positive = neighbour leads Mooloolaba)",
        ylabel="Pearson r vs Mooloolaba hsig_m",
        title="Lagged cross-correlation ±24 h (overlap window)",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "cross_source_full.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 5 — Predictive horizon heatmap
# --------------------------------------------------------------------------- #
def plot_predictive_heatmap(buoys: dict[str, pd.DataFrame]) -> None:
    start = max(df.index.min() for df in buoys.values())
    end   = min(df.index.max() for df in buoys.values())

    df_all = pd.DataFrame({
        name: buoys[name].loc[start:end, "hsig_m"] for name in BUOY_FILES
    })

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax, grid = viz.feature_horizon_heatmap(
        df_all,
        target_col="mooloolaba",
        feature_cols=NEIGHBOURS,
        horizons_h=(0, 1, 3, 6, 12, 24, 48),
        sampling_freq_min=30,
        ax=ax,
    )
    plt.tight_layout()
    out = FIG_DIR / "neighbour_predictive_full.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)

    print("\n  Pearson r (neighbour@t vs Mooloolaba@t+h):")
    print(grid.round(3).to_string())
    print(f"Saved {out.name}")


def main() -> None:
    buoys = load_all()

    print("\n--- Figure 1: coverage ---")
    plot_coverage(buoys)

    print("\n--- Figure 2: distributions & annual trends ---")
    plot_distributions_and_trends(buoys)

    print("\n--- Figure 3: seasonality & direction ---")
    plot_seasonality_and_direction(buoys)

    print("\n--- Figure 4: cross-source correlation ---")
    plot_cross_source(buoys)

    print("\n--- Figure 5: predictive horizon heatmap ---")
    plot_predictive_heatmap(buoys)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
