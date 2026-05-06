"""Tests for residual-mode training in the sequence forecasters."""
import numpy as np
import pytest

from forecast.neural import GRUForecaster


def _finite_mask(X, y):
    return ~(X.isna().any(axis=1) | y.isna())


class TestResidualMode:
    def test_constant_residual_raises(self, synthetic_df):
        """residual=True on y==X[target_col] gives zero-variance residual → ValueError."""
        df = synthetic_df(80)
        y = df["hsig_m"]  # residual is exactly zero everywhere
        model = GRUForecaster(residual=True, seq_len=4, epochs=1, device="cpu")
        with pytest.raises(ValueError, match="zero variance"):
            model.fit(df, y)

    def test_predictions_in_absolute_units(self, split):
        """Predictions from residual mode should be in hsig_m range, not near zero."""
        _, Xtr, Xte, ytr, yte = split
        mask_tr = _finite_mask(Xtr, ytr)
        mask_te = _finite_mask(Xte, yte)
        model = GRUForecaster(residual=True, seq_len=4, epochs=2, device="cpu")
        model.fit(Xtr.loc[mask_tr], ytr.loc[mask_tr])
        preds = model.predict(Xte.loc[mask_te])
        assert np.nanmean(preds) > 0.5, "Predictions look like residuals, not absolute values"

    def test_residual_y_mean_differs_from_absolute(self, split):
        """_y_mean is computed on the residual, not the absolute target, when residual=True."""
        _, Xtr, Xte, ytr, yte = split
        mask_tr = _finite_mask(Xtr, ytr)

        model_abs = GRUForecaster(residual=False, seq_len=4, epochs=1, device="cpu")
        model_abs.fit(Xtr.loc[mask_tr], ytr.loc[mask_tr])

        model_res = GRUForecaster(residual=True, seq_len=4, epochs=1, device="cpu")
        model_res.fit(Xtr.loc[mask_tr], ytr.loc[mask_tr])

        expected_residual_mean = float(
            (ytr.loc[mask_tr] - Xtr.loc[mask_tr]["hsig_m"]).mean()
        )
        np.testing.assert_allclose(model_res._y_mean, expected_residual_mean, rtol=1e-4)
        assert model_abs._y_mean != pytest.approx(model_res._y_mean, abs=0.01)
