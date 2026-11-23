"""Модели, подбор параметров и оценка прогноза продаж."""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


def load_sales(file, freq):
    if freq == "D":
        data = pd.read_csv(file, parse_dates=["date"], index_col="date")
        sales = data.loc[~data["boundary_day"], "units_recorded"]
    else:
        data = pd.read_csv(file, parse_dates=["period_start"], index_col="period_start")
        sales = data.loc[data["complete"], "units"]
    expected = pd.date_range(sales.index[0], sales.index[-1], freq=freq)
    if not sales.index.equals(expected) or sales.isna().any() or (sales < 0).any():
        raise ValueError(f"Проверьте даты и значения в {file}")
    return sales.asfreq(freq).astype(float)


def naive(train, future):
    period = 7 if train.index.freqstr == "D" else 1
    return np.resize(train.iloc[-period:].to_numpy(), len(future))


def smoothing(train, future, window=None, trend=False):
    daily = train.index.freqstr == "D"
    recent = train.iloc[-window:] if window else train
    if daily or trend:
        model = ExponentialSmoothing(
            recent, trend="add" if trend else None, damped_trend=trend,
            seasonal="add" if daily else None, seasonal_periods=7 if daily else None,
        )
    else:
        model = SimpleExpSmoothing(recent)
    return model.fit().forecast(len(future))


def autoregression(train, future, lags=None, window=None, log=False):
    daily = train.index.freqstr == "D"
    recent = train.iloc[-window:] if window else train
    target = np.log1p(recent) if log else recent
    model = AutoReg(
        target, lags=lags or ([1, 7] if daily else [1]),
        seasonal=daily, period=7 if daily else None,
    )
    forecast = model.fit().forecast(len(future))
    return np.expm1(forecast) if log else forecast


def weekdays(index):
    return np.eye(7)[index.dayofweek]


def arima(train, future, order, calendar=False):
    scale = max(1.0, train.mean())
    # Понедельник служит базой для индикаторов дня недели.
    exog = weekdays(train.index)[:, 1:] if calendar else None
    future_exog = weekdays(future)[:, 1:] if calendar else None
    model = ARIMA(train / scale, order=order, exog=exog)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fitted = model.fit(method_kwargs={"maxiter": 150})
    if not fitted.mle_retvals["converged"]:
        fitted = model.fit(method_kwargs={"maxiter": 150, "method": "powell"})
    return fitted.forecast(len(future), exog=future_exog) * scale


def boost_features(history, date, lags, windows, daily):
    row = [history[-lag] for lag in lags]
    row.extend(np.mean(history[-window:]) for window in windows)
    if daily:
        row.extend(int(date.dayofweek == day) for day in range(7))
    return row


def boosting(train, future, n_estimators, max_depth, min_samples_leaf, loss="absolute_error"):
    lags, windows = {
        "D": ([1, 7, 14, 28], [7, 28]),
        "W-MON": ([1, 2, 4], [4, 8]),
        "MS": ([1, 2], [2]),
    }[train.index.freqstr]
    daily = train.index.freqstr == "D"
    history = train.to_list()
    start = max(max(lags), max(windows))
    features = [
        boost_features(history[:i], train.index[i], lags, windows, daily)
        for i in range(start, len(train))
    ]
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        min_samples_leaf=min_samples_leaf, loss=loss, random_state=42,
    )
    model.fit(features, history[start:])

    forecast = []
    for date in future:
        row = boost_features(history, date, lags, windows, daily)
        value = max(0.0, model.predict([row])[0])
        forecast.append(value)
        history.append(value)
    return forecast


BASELINE = "AutoReg из первой версии"
MODELS = {
    "Наивная": naive,
    "Сглаживание": smoothing,
    BASELINE: autoregression,
    "AutoReg": autoregression,
    "ARIMA": arima,
    "Бустинг": boosting,
}

CONFIGS = {
    "D": {
        "Сглаживание": [{}, {"window": 84}, {"window": 112, "trend": True}],
        "AutoReg": [
            {}, {"window": 84}, {"lags": [1, 2, 7, 14], "window": 112},
            {"lags": [1, 7, 14], "log": True},
        ],
        "ARIMA": [
            {"order": (1, 0, 0), "calendar": True},
            {"order": (2, 0, 0), "calendar": True},
            {"order": (1, 1, 0), "calendar": True},
        ],
        "Бустинг": [
            {"n_estimators": 150, "max_depth": 2, "min_samples_leaf": 8},
            {"n_estimators": 300, "max_depth": 2, "min_samples_leaf": 10},
            {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 10, "loss": "huber"},
        ],
    },
    "W-MON": {
        "Сглаживание": [{}, {"trend": True}],
        "AutoReg": [{}, {"lags": [1, 2]}, {"window": 16}],
        "ARIMA": [{"order": (1, 0, 0)}, {"order": (1, 1, 0)}, {"order": (0, 1, 1)}],
        "Бустинг": [
            {"n_estimators": 150, "max_depth": 1, "min_samples_leaf": 3},
            {"n_estimators": 300, "max_depth": 2, "min_samples_leaf": 3},
        ],
    },
    "MS": {
        "AutoReg": [{"log": True}],
        "ARIMA": [{"order": (1, 0, 0)}],
        "Бустинг": [{"n_estimators": 150, "max_depth": 1, "min_samples_leaf": 2}],
    },
}


def forecast_paths(sales, label, configs, first_origin, max_horizon, stride=1):
    parts = []
    for i in range(first_origin, len(sales), stride):
        train = sales.iloc[:i]
        actual = sales.iloc[i:i + max_horizon]
        for name, params in configs.items():
            forecast = np.maximum(0, np.asarray(MODELS[name](train, actual.index, **params)))
            if forecast.shape != actual.shape or not np.isfinite(forecast).all():
                raise ValueError(f"Ошибка прогноза: {label}, {actual.index[0]}, {name}")
            parts.append(pd.DataFrame({
                "frequency": label, "model": name, "cutoff": actual.index[0],
                "date": actual.index, "step": np.arange(1, len(actual) + 1),
                "actual": actual.to_numpy(), "forecast": forecast,
            }))
    return pd.concat(parts, ignore_index=True)


def score_paths(forecasts, horizons):
    rows = []
    for name, group in forecasts.groupby("model", sort=False):
        for horizon in horizons:
            origins = group.loc[group["step"] == horizon, "cutoff"]
            path = group.loc[group["cutoff"].isin(origins) & (group["step"] <= horizon)]
            if path.empty or path["actual"].sum() <= 0:
                raise ValueError(f"Недостаточно данных для горизонта {horizon}")
            error = path["forecast"] - path["actual"]
            rows.append({
                "frequency": path["frequency"].iloc[0], "horizon": horizon,
                "model": name, "origins": len(origins),
                "MAE": error.abs().mean(), "RMSE": np.sqrt((error ** 2).mean()),
                "WAPE_pct": 100 * error.abs().sum() / path["actual"].sum(),
                "MAE_at_h": error.loc[path["step"] == horizon].abs().mean(),
            })
    return pd.DataFrame(rows).sort_values(["horizon", "WAPE_pct"])


def select_configs(train, label, horizons, validation_size, stride):
    """Подбираем параметры только на участке до начала теста."""
    chosen = {}
    records = []
    grid = CONFIGS[train.index.freqstr]
    for name in MODELS:
        candidates = grid.get(name, [{}])
        errors = []
        if len(candidates) > 1:
            for params in candidates:
                paths = forecast_paths(
                    train, label, {name: params},
                    len(train) - validation_size, max(horizons), stride,
                )
                errors.append(score_paths(paths, horizons)["WAPE_pct"].mean())
        best = int(np.argmin(errors)) if errors else 0
        chosen[name] = candidates[best]
        for i, params in enumerate(candidates):
            records.append({
                "frequency": label, "model": name, "params": json.dumps(params),
                "validation_WAPE_pct": errors[i] if errors else np.nan,
                "selected": i == best,
            })
    return chosen, pd.DataFrame(records)
