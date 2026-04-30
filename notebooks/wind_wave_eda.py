"""Wave + Wind EDA: autocorrelation, feature-horizon heatmap, direction roses, joint distributions.

Run:  ./.venv/bin/python notebooks/wind_wave_eda.py

Saves seven PNGs to notebooks/figures/:
  wind_wave_timeseries.png       — 3-panel: hsig_m, wind_speed_ms (overlaid stations), tp_s over 2020
  autocorrelation_curves.png     — ACF for wave channels and wind speed (per station)
  wind_wave_feature_horizon.png  — corr(wave+wind features at t, hsig_m at t+h) heatmap
  hsig_lookback_horizon.png      — corr(hsig_m at t-lookback, hsig_m at t+h) heatmap
  direction_roses.png            — wind direction rose per station + swell peak direction rose
  wind_wave_joint.png            — hexbin: wind speed vs wave height and wave period (primary station)
  wind_station_comparison.png    — overlaid time series + hexbin between the two stations
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz
from forecast.features import encode_circular

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Registry of available wind stations and the CSV each was downloaded to via
# `python -m qld_ckan wind --station <slug>`. Add new stations here once their CSV
# has been produced.
WIND_FILES = {
    "mountain-creek": "mountain-creek_wind_data_2015-2024.csv",
    "deception-bay":  "deception-bay_wind_data_2015-2024.csv",
}

# Stations to load for this run (order matters: the first is treated as the
# primary station for any single-station figures).
STATIONS = ["mountain-creek", "deception-bay"]

_STATION_COLORS = {
    "mountain-creek": "#e07b39",
    "deception-bay":  "#3b8db4",
}


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    wave = pd.read_csv(
        DATA_DIR / "mooloolaba_wave_data_2015-2025.csv",
        parse_dates=["datetime_utc"],
        index_col="datetime_utc",
    )
    winds: dict[str, pd.DataFrame] = {}
    for name in STATIONS:
        if name not in WIND_FILES:
            raise ValueError(f"Unknown wind station {name!r}; supported: {list(WIND_FILES)}")
        winds[name] = pd.read_csv(
            DATA_DIR / WIND_FILES[name],
            parse_dates=["datetime_utc"],
            index_col="datetime_utc",
        )
    return wave, winds


def aligned_hourly(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Resample 30-min wave data to 1h and inner-join with each station's wind.

    Wind columns are prefixed with the station slug (e.g.
    ``mountain-creek_wind_speed_ms``) so multi-station merges keep them
    distinct. The join is inner across all sources.
    """
    wave_h = wave.resample("1h").mean()
    prefixed = [w.add_prefix(f"{name}_") for name, w in winds.items()]
    return pd.concat([wave_h, *prefixed], axis=1, join="inner")


# --------------------------------------------------------------------------- #
# Figure 1 — 3-panel time series (a representative year)
# --------------------------------------------------------------------------- #
def plot_timeseries(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    year_wave = wave.loc["2020"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    viz.plot_series(
        year_wave["hsig_m"], ylabel="hsig_m (m)",
        title="Significant wave height — Mooloolaba 2020", ax=axes[0],
    )
    for name, wind in winds.items():
        year_wind = wind.loc["2020"]
        axes[1].plot(year_wind.index, year_wind["wind_speed_ms"],
                     label=name, color=_STATION_COLORS.get(name), alpha=0.75, linewidth=0.8)
    axes[1].set(ylabel="Wind speed (m/s)", title="Wind speed — 2020")
    axes[1].legend(loc="upper right", fontsize=9)
    viz.plot_series(
        year_wave["tp_s"], ylabel="tp_s (s)",
        title="Peak wave period — Mooloolaba 2020", ax=axes[2], color="#2ca02c",
    )
    fig.tight_layout()
    out = FIG_DIR / "wind_wave_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 2 — Autocorrelation curves (wave channels + wind, per station)
# --------------------------------------------------------------------------- #
def plot_autocorrelation(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    n_wave_panels = 3
    n_panels = n_wave_panels + len(winds)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    channels = [
        (wave["hsig_m"].dropna(),  30, "hsig_m (Mooloolaba)"),
        (wave["tp_s"].dropna(),    30, "tp_s (Mooloolaba)"),
        (wave["hmax_m"].dropna(),  30, "hmax_m (Mooloolaba)"),
    ]
    for name, wind in winds.items():
        channels.append((wind["wind_speed_ms"].dropna(), 60, f"wind_speed_ms ({name})"))

    for ax, (series, freq, label) in zip(axes_flat, channels):
        viz.autocorrelation_curve(
            series, max_hours=72, step_hours=1.0, sampling_freq_min=freq,
            highlight_hours=[12], threshold=0.5, source_label=label, ax=ax,
        )
    # Hide any unused subplots
    for ax in list(axes_flat)[len(channels):]:
        ax.set_visible(False)

    fig.suptitle("Autocorrelation vs lag — 72-h window, 12-h forecast horizon marked", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "autocorrelation_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 3 — Feature × horizon heatmap (wave + wind features per station)
# --------------------------------------------------------------------------- #
def plot_feature_horizon(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    df = aligned_hourly(wave, winds)

    df = encode_circular(df, periods={"peak_dir_deg": 360.0})
    for name in winds:
        col = f"{name}_wind_dir_deg"
        df[f"{name}_wind_dir_sin"] = np.sin(2 * np.pi * df[col] / 360.0)
        df[f"{name}_wind_dir_cos"] = np.cos(2 * np.pi * df[col] / 360.0)
        df = df.drop(columns=[col])

    feature_cols = [
        "hmax_m", "tz_s", "tp_s",
        "peak_dir_deg_sin", "peak_dir_deg_cos",
        "sst_c",
    ]
    for name in winds:
        feature_cols += [
            f"{name}_wind_speed_ms",
            f"{name}_wind_speed_std_ms",
            f"{name}_wind_sigma_theta_deg",
            f"{name}_wind_dir_sin",
            f"{name}_wind_dir_cos",
        ]

    fig, ax = plt.subplots(figsize=(13, 0.45 * len(feature_cols) + 3))
    ax, grid = viz.feature_horizon_heatmap(
        df,
        target_col="hsig_m",
        feature_cols=feature_cols,
        horizons_h=(1, 3, 6, 12, 24, 48, 72),
        sampling_freq_min=60,
        source_label="Mooloolaba hsig_m target",
        ax=ax,
    )
    fig.tight_layout()
    out = FIG_DIR / "wind_wave_feature_horizon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")
    print("  Feature-horizon grid:")
    print(grid.round(3).to_string())


# --------------------------------------------------------------------------- #
# Figure 4 — Lookback × horizon heatmap for hsig_m
# --------------------------------------------------------------------------- #
def plot_lookback_horizon(wave: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax, grid = viz.lookback_horizon_heatmap(
        wave["hsig_m"].dropna(),
        lookbacks_h=(0, 0.5, 1, 3, 6, 12, 24, 48),
        horizons_h=(0, 1, 3, 6, 12, 24, 48, 72),
        sampling_freq_min=30,
        source_label="Mooloolaba hsig_m",
        ax=ax,
    )
    fig.tight_layout()
    out = FIG_DIR / "hsig_lookback_horizon.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 5 — Direction roses: one wind rose per station + swell rose
# --------------------------------------------------------------------------- #
def plot_direction_roses(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    bin_edges = np.linspace(0, 2 * np.pi, 37)   # 10° bins
    bin_width = bin_edges[1] - bin_edges[0]

    panels = [
        (name, wind["wind_dir_deg"].dropna(), f"Wind direction\n{name}",
         _STATION_COLORS.get(name, "#888"))
        for name, wind in winds.items()
    ]
    panels.append(("swell", wave["peak_dir_deg"].dropna(),
                   "Peak swell direction\nMooloolaba", "#1f77b4"))

    n = len(panels)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (_, series, title, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, n, i, projection="polar")
        dirs_rad = np.deg2rad(series.values)
        counts, _ = np.histogram(dirs_rad, bins=bin_edges)
        fracs = counts / counts.sum()
        ax.bar(bin_edges[:-1], fracs, width=bin_width, alpha=0.75, color=color)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(title, pad=14)
        ax.yaxis.set_visible(False)

    fig.suptitle("Wind vs swell peak direction distributions  (all years pooled)", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "direction_roses.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 6 — Wind-wave joint distributions (primary station only)
# --------------------------------------------------------------------------- #
def plot_wind_wave_joint(wave: pd.DataFrame, winds: dict[str, pd.DataFrame]) -> None:
    primary = next(iter(winds))
    df = aligned_hourly(wave, {primary: winds[primary]}).dropna(
        subset=["hsig_m", f"{primary}_wind_speed_ms", "tp_s"]
    )
    speed = df[f"{primary}_wind_speed_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hb = axes[0].hexbin(speed, df["hsig_m"], gridsize=45, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=axes[0], label="Count")
    axes[0].set(
        xlabel="Wind speed (m/s)", ylabel="hsig_m (m)",
        title="Wind speed vs significant wave height",
    )

    hb2 = axes[1].hexbin(speed, df["tp_s"], gridsize=45, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb2, ax=axes[1], label="Count")
    axes[1].set(
        xlabel="Wind speed (m/s)", ylabel="tp_s (s)",
        title="Wind speed vs peak wave period",
    )

    fig.suptitle(
        f"Wave–wind joint distributions  (Mooloolaba + {primary}, aligned to hourly)",
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG_DIR / "wind_wave_joint.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


# --------------------------------------------------------------------------- #
# Figure 7 — Cross-station comparison (overlaid time series + hexbin)
# --------------------------------------------------------------------------- #
def plot_station_comparison(winds: dict[str, pd.DataFrame]) -> None:
    if len(winds) < 2:
        print("Skipping station comparison: only one station loaded.")
        return

    names = list(winds)
    a, b = names[0], names[1]
    speed_a = winds[a]["wind_speed_ms"]
    speed_b = winds[b]["wind_speed_ms"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    year_a = speed_a.loc["2020"]
    year_b = speed_b.loc["2020"]
    axes[0].plot(year_a.index, year_a, label=a,
                 color=_STATION_COLORS.get(a), alpha=0.75, linewidth=0.8)
    axes[0].plot(year_b.index, year_b, label=b,
                 color=_STATION_COLORS.get(b), alpha=0.75, linewidth=0.8)
    axes[0].set(xlabel="Time", ylabel="Wind speed (m/s)", title="Wind speed — 2020")
    axes[0].legend(loc="upper right")

    joint = pd.concat([speed_a, speed_b], axis=1, join="inner",
                      keys=[a, b]).dropna()
    r = joint[a].corr(joint[b])
    hb = axes[1].hexbin(joint[a], joint[b], gridsize=45, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=axes[1], label="Count")
    lim = max(joint[a].max(), joint[b].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.5)
    axes[1].set(
        xlabel=f"{a} wind speed (m/s)", ylabel=f"{b} wind speed (m/s)",
        title=f"Hourly speed agreement  (Pearson r = {r:.3f})",
        xlim=(0, lim), ylim=(0, lim),
    )

    fig.suptitle("Wind station comparison", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "wind_station_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.name}")


def main() -> None:
    wave, winds = load_data()
    print(f"Wave: {len(wave):,} rows  ({wave.index.min().date()} → {wave.index.max().date()})")
    for name, wind in winds.items():
        print(f"Wind ({name}): {len(wind):,} rows  ({wind.index.min().date()} → {wind.index.max().date()})")

    print("\n--- Figure 1: wave + wind time series (2020) ---")
    plot_timeseries(wave, winds)

    print("\n--- Figure 2: autocorrelation curves ---")
    plot_autocorrelation(wave, winds)

    print("\n--- Figure 3: feature × horizon heatmap ---")
    plot_feature_horizon(wave, winds)

    print("\n--- Figure 4: hsig_m lookback × horizon heatmap ---")
    plot_lookback_horizon(wave)

    print("\n--- Figure 5: direction roses ---")
    plot_direction_roses(wave, winds)

    print("\n--- Figure 6: wind-wave joint distributions ---")
    plot_wind_wave_joint(wave, winds)

    print("\n--- Figure 7: station comparison ---")
    plot_station_comparison(winds)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
