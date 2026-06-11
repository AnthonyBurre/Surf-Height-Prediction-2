import numpy as np
import pandas as pd
import pytest

from forecast import models


def _xy(n=500, p=5):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(X["f0"] * 2 - X["f1"] + rng.normal(0, 0.1, n))
    return X, y


def test_predict_returns_series_indexed_like_X():
    X, y = _xy()
    m = models.RidgeForecaster(alpha=1.0).fit(X, y)
    pred = m.predict(X)
    assert isinstance(pred, pd.Series)
    assert pred.index.equals(X.index)


def test_schema_enforced_missing_raises_extra_dropped():
    X, y = _xy()
    m = models.RidgeForecaster().fit(X, y)
    with pytest.raises(ValueError):
        m.predict(X.drop(columns=["f0"]))            # missing required -> error
    extra = X.copy()
    extra["unexpected"] = 1.0
    pred = m.predict(extra)                            # extra col dropped, ordered
    assert pred.notna().all()


def test_hgb_handles_nan_natively():
    X, y = _xy()
    Xn = X.copy()
    Xn.iloc[0, 0] = np.nan
    m = models.HGBForecaster(max_iter=30).fit(Xn, y)
    assert m.predict(Xn).notna().all()


def test_direct_multi_horizon_fits_one_model_per_horizon():
    X, y = _xy()
    dm = models.DirectMultiHorizon(lambda h, s: models.RidgeForecaster(), horizons=(6, 12))
    dm.fit({6: (X, y), 12: (X, y)})
    preds = dm.predict({6: X, 12: X})
    assert set(preds) == {6, 12}
    assert all(isinstance(v, pd.Series) for v in preds.values())
