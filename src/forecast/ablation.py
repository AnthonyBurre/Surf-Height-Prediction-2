"""Feature-ablation analysis — the keep-if-helps-or-hurts selection rule.

The ablation *sweep* lives in ``notebooks/feature_ablation.py``: it loads every
source once via ``forecast.load_all_sources`` (a ``SourceBundle``) and builds
each subset's design matrix via ``forecast.build_design`` — both now
general-purpose core helpers in ``forecast.data`` / ``forecast.features``,
since "load these sources" and "build a design from them" are not
ablation-specific.

What remains here is the one genuinely ablation-specific step: turning the
long-form add-one/drop-one result table into a per-(family, horizon) station
recommendation.
"""
import pandas as pd


def recommended_set(
    report_df: pd.DataFrame,
    threshold: float = 0.005,
) -> dict[tuple[str, int], list[str]]:
    """Apply the keep-if-helps-or-hurts rule per (family, horizon).

    A station is kept when EITHER:
      - add-one RMSE improves the primary-only baseline by ≥ ``threshold``
        (relative), OR
      - drop-one RMSE worsens the ceiling by ≥ ``threshold`` (relative).

    Args:
        report_df: long-form rows with columns ``family``, ``horizon_h``,
            ``direction`` (one of ``baseline``/``ceiling``/``add``/``drop``),
            ``station`` (None for baseline/ceiling), ``RMSE``.
        threshold: relative-RMSE threshold (default 0.5%).

    Returns:
        dict keyed by (family, horizon_h) → list of station slugs to keep
        (sorted alphabetically for stable display).
    """
    out: dict[tuple[str, int], list[str]] = {}
    for (family, h), grp in report_df.groupby(["family", "horizon_h"]):
        base = grp[grp["direction"] == "baseline"]["RMSE"]
        ceil = grp[grp["direction"] == "ceiling"]["RMSE"]
        if base.empty or ceil.empty:
            continue
        base_rmse = float(base.iloc[0])
        ceil_rmse = float(ceil.iloc[0])
        keep: list[str] = []
        adds = grp[grp["direction"] == "add"].set_index("station")["RMSE"]
        drops = grp[grp["direction"] == "drop"].set_index("station")["RMSE"]
        stations = sorted(set(adds.index) | set(drops.index))
        for s in stations:
            add_gain = (base_rmse - float(adds[s])) / base_rmse if s in adds else 0.0
            drop_cost = (float(drops[s]) - ceil_rmse) / ceil_rmse if s in drops else 0.0
            if add_gain >= threshold or drop_cost >= threshold:
                keep.append(s)
        out[(family, int(h))] = keep
    return out
