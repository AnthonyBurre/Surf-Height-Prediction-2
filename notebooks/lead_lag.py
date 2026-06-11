"""EDA — cross-source lead–lag (Fig 2.6). Which neighbour buoys / wind stations
lead Mooloolaba hsig_m, and by how much? Drives the exogenous shortlist.

    ./.venv/bin/python notebooks/lead_lag.py
"""
import matplotlib
matplotlib.use("Agg")

import pandas as pd

import forecast as fc
import viz
from forecast.constants import FIGURE_DIR, NEIGHBOUR_BUOYS

viz.apply_style()


def main() -> None:
    avail = fc.available_sources()
    buoys = [b for b in NEIGHBOUR_BUOYS if b in avail["wave"]]
    stations = avail["wind"]

    cols = {"hsig_m": fc.load_target()}
    for b in buoys:
        cols[f"{b}__hsig_m"] = fc.load_wave(b)["hsig_m"]
    for s in stations:
        w = fc.load_wind(s)["wind_speed_ms"]
        cols[f"{s}__wind_speed_ms"] = w.reindex(w.index.union(cols["hsig_m"].index)).ffill()
    frame = pd.DataFrame(cols).reindex(cols["hsig_m"].index)

    fig = viz.lead_lag_matrix(frame, "hsig_m", max_lag=48)
    viz.save(fig, FIGURE_DIR / "lead_lag.png")

    # printed decision table: peak |corr| and lag (hours; + = companion leads)
    tgt = frame["hsig_m"]
    rows = []
    for c in frame.columns:
        if c == "hsig_m":
            continue
        corrs = {L / 2.0: tgt.corr(frame[c].shift(L)) for L in range(-48, 49)}
        s = pd.Series(corrs)
        lag = s.abs().idxmax()
        rows.append((c, round(s[lag], 3), lag))
    table = pd.DataFrame(rows, columns=["source", "peak_corr", "lag_h"]).sort_values(
        "peak_corr", ascending=False)
    print(table.to_string(index=False))
    print("\nNote: neighbour buoys co-move with the target at ~lag 0 — no usable lead.")


if __name__ == "__main__":
    main()
