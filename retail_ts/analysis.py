"""Descriptive, robust and stationarity diagnostics with explicit assumptions."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


def validate_series(series: pd.Series, min_length: int = 3) -> pd.Series:
    """Reject missing, infinite, unsorted, duplicate or irregular time series."""
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Expected a Series with a DatetimeIndex")
    if len(series) < min_length:
        raise ValueError(f"At least {min_length} observations are required")
    if not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("Time index must be sorted and unique")
    if pd.infer_freq(series.index) is None:
        raise ValueError("An equally spaced calendar series is required; do not drop interior dates")
    x = series.astype(float)
    if not np.isfinite(x).all():
        raise ValueError("Series contains missing or infinite observations")
    return x


def describe_series(series: pd.Series) -> pd.Series:
    """Summarize volume, dispersion and intermittency; CV only for positive mean."""
    x = validate_series(series)
    mean, std = x.mean(), x.std(ddof=1)
    positive = x[x.gt(0)]
    return pd.Series({"n": len(x), "sum": x.sum(), "mean": mean, "median": x.median(),
                      "std": std, "variance": x.var(ddof=1), "min": x.min(),
                      "q25": x.quantile(.25), "q75": x.quantile(.75),
                      "p95": x.quantile(.95), "max": x.max(),
                      "zero_share": x.eq(0).mean(), "cv": std / mean if mean > 0 else np.nan,
                      "skew": x.skew(), "excess_kurtosis": x.kurt(),
                      "dispersion_index": x.var(ddof=1) / mean if mean > 0 else np.nan,
                      "adi": len(x) / len(positive) if len(positive) else np.nan,
                      "cv2_positive": (positive.std() / positive.mean()) ** 2
                      if len(positive) > 1 else np.nan})


def outlier_diagnostics(series: pd.Series, threshold: float = 3.5) -> pd.DataFrame:
    """Flag IQR and MAD candidates; never remove observations.

    Global rules are screening tools, not probabilities of measurement error.
    A zero MAD produces an undefined score rather than division by zero.
    """
    x = validate_series(series)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    q1, q3 = x.quantile([.25, .75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    mad = np.median(np.abs(x - x.median()))
    score = .67448975 * (x - x.median()) / mad if mad > 0 else pd.Series(np.nan, index=x.index)
    return pd.DataFrame({"units": x, "iqr_lower": lower, "iqr_upper": upper,
                         "robust_z": score, "iqr_flag": (x < lower) | (x > upper),
                         "mad_flag": np.abs(score) > threshold})


def decompose_stl(series: pd.Series, period: int = 7,
                  seasonal: int = 13, trend: int = 29) -> tuple[pd.DataFrame, pd.Series]:
    """Fit robust additive STL and return descriptive component strengths.

    At least two complete cycles are required. Strength is scale-dependent and
    is not a significance test or a fraction of explained variance in an OLS fit.
    """
    if not isinstance(period, int) or period < 2:
        raise ValueError("period must be an integer >= 2")
    if seasonal < 7 or seasonal % 2 == 0 or trend <= period or trend % 2 == 0:
        raise ValueError("Use odd seasonal >= 7 and odd trend > period")
    x = validate_series(series, min_length=2 * period)
    fit = STL(x, period=period, seasonal=seasonal, trend=trend, robust=True).fit()
    components = pd.DataFrame({"observed": x, "trend": fit.trend,
                               "seasonal": fit.seasonal, "residual": fit.resid})
    def strength(component: pd.Series) -> float:
        denominator = np.var(component + fit.resid, ddof=1)
        return float(max(0, 1 - np.var(fit.resid, ddof=1) / denominator)) if denominator > 0 else np.nan
    metrics = pd.Series({"seasonal_strength": strength(fit.seasonal),
                         "trend_strength": strength(fit.trend),
                         "cycles": len(x) / period,
                         "residual_acf1": acf(fit.resid, nlags=1, fft=False)[1],
                         "residual_acf7": acf(fit.resid, nlags=7, fft=False)[7]})
    return components, metrics


def stationarity_tests(series: pd.Series, label: str,
                       regression: str = "c", alpha: float = .05,
                       adf_maxlag: int = 21,
                       kpss_lags: str | int = "auto") -> pd.DataFrame:
    """Run complementary ADF and KPSS tests, preserving tabulated p bounds.

    ADF: lag count selected by AIC within the stated maximum. KPSS: automatic
    bandwidth or explicit sensitivity bandwidth. A boundary p-value is a bound,
    not an exact probability. Non-rejection does not prove the null hypothesis.
    """
    if regression not in {"c", "ct"} or not 0 < alpha < 1:
        raise ValueError("regression must be c/ct and alpha must be between 0 and 1")
    x = validate_series(series, min_length=30)
    if x.nunique() < 2:
        raise ValueError("Unit-root tests are undefined for a constant series")
    if adf_maxlag < 0:
        raise ValueError("adf_maxlag must be nonnegative")
    maxlag = min(adf_maxlag, len(x) // 2 - (2 if regression == "ct" else 1) - 1)
    a = adfuller(x, regression=regression, maxlag=maxlag, autolag="AIC")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        k = kpss(x, regression=regression, nlags=kpss_lags)
    interpolation = any(issubclass(w.category, InterpolationWarning) for w in caught)
    relation = "≤" if interpolation and k[1] == .01 else "≥" if interpolation and k[1] == .1 else "="
    base = {"series": label, "regression": regression, "alpha": alpha, "n_input": len(x)}
    return pd.DataFrame([
        {**base, "test": "ADF", "statistic": a[0], "p_value": a[1], "p_relation": "=",
         "p_display": f"{a[1]:.6g}", "lags": a[2], "n_effective": a[3],
         "critical_5pct": a[4]["5%"], "reject_H0": bool(a[1] < alpha),
         "lag_method": f"AIC, maxlag={maxlag}", "warning": ""},
        {**base, "test": "KPSS", "statistic": k[0], "p_value": k[1], "p_relation": relation,
         "p_display": f"{relation} {k[1]:.6g}", "lags": k[2], "n_effective": len(x),
         "critical_5pct": k[3]["5%"], "reject_H0": bool(k[1] < alpha),
         "lag_method": str(kpss_lags), "warning": " | ".join(str(w.message) for w in caught)}])


def transform_series(series: pd.Series, *, log: bool = False,
                     difference: int = 0) -> pd.Series:
    """Apply log1p (nonnegative input only) and at most one stated difference.

    difference=7 compares the same weekday, difference=1 consecutive days.
    Missing leading values are removed only after validating the original grid.
    """
    x = validate_series(series)
    if not isinstance(difference, int) or difference < 0 or difference >= len(x) - 2:
        raise ValueError("difference must be a nonnegative lag leaving >= 3 observations")
    if log:
        if x.lt(0).any():
            raise ValueError("Sales log1p requires nonnegative units")
        x = np.log1p(x)
    if difference:
        x = x.diff(difference).iloc[difference:]
    return x


def calendar_profile(series: pd.Series) -> pd.DataFrame:
    """Compare weekday level and dispersion without an independence assumption."""
    x = validate_series(series)
    return x.groupby(x.index.dayofweek).agg(n="size", mean="mean", median="median",
                                           std="std", zero_share=lambda y: y.eq(0).mean())


def autocorrelation_summary(series: pd.Series, lags: tuple[int, ...] = (1, 7, 14, 21)) -> pd.Series:
    """Return selected ACF coefficients as descriptive seasonal evidence."""
    x = validate_series(series)
    if not lags or min(lags) < 1 or max(lags) >= len(x):
        raise ValueError("lags must be positive and smaller than series length")
    values = acf(x, nlags=max(lags), fft=False)
    return pd.Series({f"acf_{lag}": values[lag] for lag in lags})
