"""Aggregate ``ablate_*`` runs into heatmaps + recommended-set tables.

Run:
    ./.venv/bin/python notebooks/feature_ablation_report.py

Reads every ``ablate_*`` row from ``experiments.jsonl``, dedupes by
(family, horizon, direction, station) keeping the most recent timestamp,
and produces:

  notebooks/figures/feature_ablation_<family>_add.png
  notebooks/figures/feature_ablation_<family>_drop.png

…plus a markdown table to stdout listing the recommended station set per
(family, horizon), which can be pasted directly into the README.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecast as fc
from forecast import ablation as ab

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Display order — wave neighbours first (geographic-ish: north → south, then
# distant), then wind stations. Stable order makes the heatmaps comparable
# across families.
DISPLAY_ORDER: list[str] = ab.STATIONS_WAVE + ab.STATIONS_WIND


def load_ablation_runs() -> pd.DataFrame:
    """Long-form table: one row per (family, horizon_h, direction, station)."""
    log = fc.find_runs(name_prefix="ablate_")
    if log.empty:
        raise RuntimeError("No ablate_* entries in experiments.jsonl. "
                           "Run notebooks/feature_ablation.py first.")
    rows = []
    for _, r in log.iterrows():
        extra = r["extra"] or {}
        metrics = r["metrics"] or {}
        rows.append({
            "family": extra.get("family"),
            "horizon_h": extra.get("horizon_h"),
            "direction": extra.get("direction"),
            "station": extra.get("station"),
            "RMSE": metrics.get("RMSE"),
            "skill": metrics.get("SkillVsBaseline"),
            "timestamp": r["timestamp"],
        })
    df = pd.DataFrame(rows).dropna(subset=["family", "horizon_h", "direction"])
    df["horizon_h"] = df["horizon_h"].astype(int)
    # Most recent row per cell wins (lets re-runs supersede old entries).
    df = (df.sort_values("timestamp")
            .drop_duplicates(["family", "horizon_h", "direction", "station"],
                             keep="last")
            .drop(columns="timestamp")
            .reset_index(drop=True))
    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Per (family, h, station) add-skill-Δ and drop-skill-Δ.

    add_delta_pct   = (baseline_RMSE - add_RMSE)   / baseline_RMSE  × 100
                       (positive = adding this station helps)
    drop_delta_pct  = (drop_RMSE   - ceiling_RMSE) / ceiling_RMSE   × 100
                       (positive = dropping this station hurts → station mattered)
    """
    base = (df[df["direction"] == "baseline"]
            .set_index(["family", "horizon_h"])["RMSE"]
            .rename("baseline_RMSE"))
    ceil = (df[df["direction"] == "ceiling"]
            .set_index(["family", "horizon_h"])["RMSE"]
            .rename("ceiling_RMSE"))
    adds = df[df["direction"] == "add"][["family", "horizon_h", "station", "RMSE"]] \
        .rename(columns={"RMSE": "add_RMSE"})
    drops = df[df["direction"] == "drop"][["family", "horizon_h", "station", "RMSE"]] \
        .rename(columns={"RMSE": "drop_RMSE"})
    merged = adds.merge(drops, on=["family", "horizon_h", "station"], how="outer")
    merged = (merged.join(base, on=["family", "horizon_h"])
                    .join(ceil, on=["family", "horizon_h"]))
    merged["add_delta_pct"] = (merged["baseline_RMSE"] - merged["add_RMSE"]) / merged["baseline_RMSE"] * 100
    merged["drop_delta_pct"] = (merged["drop_RMSE"] - merged["ceiling_RMSE"]) / merged["ceiling_RMSE"] * 100
    return merged


def _heatmap_panel(ax, matrix: pd.DataFrame, title: str, cbar_label: str,
                   cmap: str = "RdBu_r", vlim: float | None = None):
    """One heatmap panel: rows = stations, cols = horizons, cell = signed-pct."""
    if vlim is None:
        vlim = max(0.5, np.nanmax(np.abs(matrix.to_numpy())))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap,
                   vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("horizon (h)")
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iat[i, j]
            if pd.isna(val):
                continue
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color="black", fontsize=7)
    ax._cbar_label = cbar_label  # stashed for the caller's colorbar
    return im


def render_heatmaps(deltas: pd.DataFrame, families: list[str]) -> None:
    """Two figures per family: add and drop heatmaps.

    Sign convention is "positive = station is contributing" for BOTH panels:
      - add panel cell = baseline_RMSE − add_RMSE  (% of baseline_RMSE);
        positive ⇒ adding this station LOWERS RMSE ⇒ station helps.
      - drop panel cell = drop_RMSE − ceiling_RMSE (% of ceiling_RMSE);
        positive ⇒ removing this station RAISES RMSE ⇒ station mattered.
    Cell values are NOT raw Δ-RMSE (which under the "new − old" reading would
    flip the sign of the add panel) — they're explicitly labelled as "RMSE %
    gain" / "RMSE % cost" so positive always means "station contributes".
    """
    horizons = sorted(deltas["horizon_h"].unique())
    for family in families:
        fam = deltas[deltas["family"] == family]
        if fam.empty:
            print(f"  [skip] no {family} rows")
            continue
        for direction, value_col, title_suffix, cbar_label in [
            ("add",  "add_delta_pct",
             "add-one  ·  RMSE % gain vs primary-only baseline  (positive = station helps)",
             "RMSE % gain"),
            ("drop", "drop_delta_pct",
             "drop-one  ·  RMSE % cost vs full ceiling  (positive = station mattered)",
             "RMSE % cost"),
        ]:
            matrix = (fam.pivot_table(index="station", columns="horizon_h",
                                      values=value_col, aggfunc="first")
                        .reindex(index=DISPLAY_ORDER, columns=horizons))
            fig, ax = plt.subplots(figsize=(7, 6))
            im = _heatmap_panel(ax, matrix,
                                title=f"{family.upper()}  ·  {title_suffix}",
                                cbar_label=cbar_label)
            fig.colorbar(im, ax=ax, label=cbar_label)
            fig.tight_layout()
            out = FIG_DIR / f"feature_ablation_{family}_{direction}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out.name}")


def print_recommended_table(df: pd.DataFrame, families: list[str], threshold: float = 0.005) -> None:
    """Print the recommended-set table as markdown ready for the README."""
    long = df.copy()
    long["horizon_h"] = long["horizon_h"].astype(int)
    recs = ab.recommended_set(long, threshold=threshold)
    horizons = sorted({h for (_, h) in recs.keys()})

    print(f"\n## Recommended station set per (horizon, model family) — threshold {threshold:.1%}\n")
    header = "| horizon | " + " | ".join(f.upper() for f in families) + " |"
    sep = "|---:|" + "|".join(["---"] * len(families)) + "|"
    print(header)
    print(sep)
    for h in horizons:
        cells = []
        for fam in families:
            stations = recs.get((fam, h))
            if stations is None:
                cells.append("—")
            elif not stations:
                cells.append("(none)")
            else:
                cells.append(", ".join(stations))
        print(f"| {h}h | " + " | ".join(cells) + " |")


def print_anchor_table(df: pd.DataFrame, families: list[str]) -> None:
    """Per-horizon baseline / ceiling RMSE + skill for each family."""
    print("\n## Anchors per (horizon, family)\n")
    horizons = sorted(df["horizon_h"].unique())
    header = "| horizon | family | baseline RMSE | ceiling RMSE | gap % | base skill | ceil skill |"
    sep = "|---:|:---|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for h in horizons:
        for fam in families:
            sub = df[(df["horizon_h"] == h) & (df["family"] == fam)]
            b = sub[sub["direction"] == "baseline"]
            c = sub[sub["direction"] == "ceiling"]
            if b.empty or c.empty:
                continue
            br, cr = float(b["RMSE"].iloc[0]), float(c["RMSE"].iloc[0])
            bs, cs = float(b["skill"].iloc[0] or 0), float(c["skill"].iloc[0] or 0)
            gap = (br - cr) / br * 100
            print(f"| {h}h | {fam} | {br:.4f} | {cr:.4f} | {gap:+.2f}% | {bs:+.3f} | {cs:+.3f} |")


def main() -> None:
    df = load_ablation_runs()
    families = sorted(df["family"].unique())
    print(f"Loaded {len(df)} ablation rows across families: {families}")

    deltas = compute_deltas(df)
    print(f"\nRendering heatmaps under {FIG_DIR}/ …")
    render_heatmaps(deltas, families)

    print_anchor_table(df, families)
    print_recommended_table(df, families)


if __name__ == "__main__":
    main()
