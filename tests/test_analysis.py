"""Guardrails for transformations, time spacing and statistical diagnostics."""
import numpy as np
import pandas as pd
import pytest

from retail_ts.analysis import (transform_series, stationarity_tests, decompose_stl,
                                outlier_diagnostics, validate_series)


def test_log_difference_matches_its_mathematical_definition():
    x = pd.Series([0, 3, 0, 7, 15, 0, 1, 3, 7, 8], index=pd.date_range("2020-01-01", periods=10))
    result = transform_series(x, log=True, difference=7)
    assert result.index.equals(x.index[7:])
    np.testing.assert_allclose(result.to_numpy(), np.log1p(x.iloc[7:].to_numpy()) - np.log1p(x.iloc[:-7].to_numpy()))
    assert x.iloc[0] == 0


def test_log_transform_rejects_negative_sales():
    x = pd.Series([1, -1, 2], index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="nonnegative"):
        transform_series(x, log=True)


def test_interior_missing_dates_are_not_silently_compressed():
    x = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10)).drop(pd.Timestamp("2020-01-05"))
    with pytest.raises(ValueError, match="equally spaced"):
        validate_series(x)


def test_missing_values_require_an_explicit_policy():
    x = pd.Series([1, np.nan, 2], index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="missing"):
        transform_series(x)


def test_constant_and_short_series_are_not_unit_root_tested():
    x = pd.Series(np.ones(40), index=pd.date_range("2020-01-01", periods=40))
    with pytest.raises(ValueError, match="constant"):
        stationarity_tests(x, "constant")
    with pytest.raises(ValueError, match="30"):
        stationarity_tests(x.iloc[:20], "short")


def test_zero_mad_is_marked_undefined_and_no_values_are_removed():
    x = pd.Series([0, 0, 0, 0, 100], index=pd.date_range("2020-01-01", periods=5))
    result = outlier_diagnostics(x)
    assert result.robust_z.isna().all()
    assert result.units.tolist() == x.tolist()
    assert result.iqr_flag.iloc[-1]


def test_stl_requires_two_cycles_and_reconstructs_input():
    x = pd.Series(np.tile([1, 2, 3, 2, 1, 0, 0], 8), index=pd.date_range("2020-01-01", periods=56))
    with pytest.raises(ValueError, match="14"):
        decompose_stl(x.iloc[:10], period=7)
    components, strength = decompose_stl(x)
    np.testing.assert_allclose(components.trend + components.seasonal + components.residual, x)
    assert 0 <= strength.seasonal_strength <= 1


def test_stationarity_results_keep_sample_sizes_and_p_value_bounds():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(size=120), index=pd.date_range("2020-01-01", periods=120))
    result = stationarity_tests(x, "fixture", adf_maxlag=14, kpss_lags=7).set_index("test")
    assert result.loc["ADF", "n_effective"] == 120 - result.loc["ADF", "lags"] - 1
    assert result.loc["KPSS", "lags"] == 7
    if result.loc["KPSS", "p_value"] == .1:
        assert result.loc["KPSS", "p_relation"] == "≥"
    assert result.alpha.eq(.05).all()
