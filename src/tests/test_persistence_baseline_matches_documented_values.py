"""Guard test: pin the persistence baseline so a future upstream data revision is
*caught* rather than silently shifting the headline numbers.

The QLD portal periodically re-derives and republishes whole years (the README
documents a re-pull that changed ~27% of rows and pushed persistence RMSE from
~26 cm to ~40 cm). Absolute RMSE is therefore not portable across data vintages,
so this test pins the *current-vintage* persistence RMSE with a tolerance wide
enough to ignore float noise but tight enough to flag a republish. Skill is the
portable headline and is reported in the README; here we assert the
persistence↔climatology crossover sits where it does now.

If this test fails after a re-download, the data was revised: re-baseline the
README numbers (don't just widen the tolerance).
"""
import numpy as np
import pandas as pd
import pytest

from forecast import baselines, data, metrics, targets

# Pinned full-series persistence RMSE (metres), current QLD vintage (2026-06).
DOCUMENTED_PERSISTENCE_RMSE = {6: 0.2724, 12: 0.3801, 24: 0.5066}
ATOL = 0.03  # ~3 cm: ignores float noise, catches a ~27%-of-rows republish


def _data_available() -> bool:
    try:
        return "mooloolaba" in data.available_sources()["wave"]
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _data_available(), reason="Mooloolaba CSV not present in data/."
)


def _persistence_rmse(y: pd.Series, h: int) -> float:
    tgt = targets.make_target(y, h)
    X = pd.DataFrame(index=tgt.dropna().index)
    pred = baselines.Persistence(y, h).predict(X)
    return metrics.rmse(tgt.reindex(X.index), pred)


def test_persistence_rmse_matches_documented_values():
    y = data.load_target()
    for h, expected in DOCUMENTED_PERSISTENCE_RMSE.items():
        got = _persistence_rmse(y, h)
        assert np.isclose(got, expected, atol=ATOL), (
            f"h={h}h persistence RMSE {got:.4f} != pinned {expected:.4f} "
            f"(atol {ATOL}). Upstream data may have been revised — re-baseline."
        )


def test_climatology_crossover_is_between_12h_and_24h():
    """Persistence beats climatology short; climatology wins by 24h (the crossover
    that tells us what skill is even available at each horizon)."""
    y = data.load_target()

    def sm_rmse(h):
        tgt = targets.make_target(y, h)
        X = pd.DataFrame(index=tgt.dropna().index)
        sm = baselines.SeasonalMean(y, h).fit(X, y.reindex(X.index))
        return metrics.rmse(tgt.reindex(X.index), sm.predict(X))

    assert _persistence_rmse(y, 12) < sm_rmse(12)   # persistence better at 12h
    assert _persistence_rmse(y, 24) > sm_rmse(24)   # climatology better at 24h
