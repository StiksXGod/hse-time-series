"""Consistently labeled figures used directly by the research notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BLUE, ORANGE, GREEN, GRAY = "#24558A", "#CB5B36", "#248779", "#858E9A"
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


def set_style() -> None:
    """Use a font with Cyrillic support and readable static report defaults."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18,
                         "figure.dpi": 110, "savefig.dpi": 140,
                         "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, GREEN, GRAY])})


def date_axis(ax: plt.Axes, ylabel: str) -> None:
    """Give a time plot a compact date scale and explicit units."""
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(xlabel="Дата", ylabel=ylabel)


def save_figure(fig: Figure, directory: Path, name: str) -> None:
    """Save the same figure that is displayed inline in the notebook."""
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", bbox_inches="tight", facecolor="white")


def plot_aggregations(daily: pd.DataFrame, weekly: pd.DataFrame,
                      monthly: pd.DataFrame) -> Figure:
    """Show all three calendars and distinguish incomplete endpoint periods."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    axes[0].plot(daily.index, daily.units_recorded, lw=1.1)
    axes[0].scatter(daily.index[daily.boundary_day], daily.units_recorded[daily.boundary_day],
                    color=ORANGE, label="Граничный день", zorder=3)
    axes[0].set_title("Дневные зарегистрированные продажи выбранного товара")
    date_axis(axes[0], "Единиц / день")
    axes[0].legend()
    for ax, table, title, unit in zip(axes[1:], [weekly, monthly],
                                    ["Недели: понедельник — воскресенье", "Календарные месяцы"],
                                    ["Единиц / неделю", "Единиц / месяц"]):
        colors = np.where(table.complete, BLUE, ORANGE)
        ax.bar(table.index, table.units, width=5 if unit.endswith("неделю") else 21,
               align="edge", color=colors, alpha=.9)
        ax.scatter([], [], color=BLUE, marker="s", label="Полный интервал")
        ax.scatter([], [], color=ORANGE, marker="s", label="Граничный / неполный")
        ax.set_title(title)
        date_axis(ax, unit)
        ax.legend(loc="upper left")
    return fig


def plot_coverage(daily: pd.DataFrame) -> Figure:
    """Expose whole-store inactivity separately from SKU zero sales."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained", sharex=True)
    axes[0].plot(daily.index, daily.store_rows, color=GRAY, lw=1)
    axes[0].set_title("Полнота календаря: все строки магазина после удаления дублей")
    date_axis(axes[0], "Строк / день")
    axes[1].plot(daily.index, daily.units_recorded, label="Зарегистрировано", lw=1)
    for mask, color, marker, name in [
        (~daily.store_observed & daily.is_saturday, GRAY, "s", "Нет записей: суббота"),
        (daily.silent_non_saturday, ORANGE, "x", "Нет записей: другие дни"),
        (daily.store_observed & daily.units_recorded.eq(0), GREEN, "o", "Магазин активен, товар = 0")]:
        axes[1].scatter(daily.index[mask], daily.units_recorded[mask], color=color,
                        marker=marker, s=26, label=name, zorder=3)
    axes[1].set_title("Нулевые продажи и отсутствие записей — разные диагностические признаки")
    date_axis(axes[1], "Единиц / день")
    axes[1].legend(fontsize=9)
    return fig


def plot_outliers(original: pd.Series, cleaned: pd.Series,
                   diagnostics: pd.DataFrame) -> Figure:
    """Compare duplicate handling and flag extremes without winsorization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained", sharex=True)
    axes[0].plot(original.index, original, color=GRAY, label="До удаления точных дублей", lw=1.8)
    axes[0].plot(cleaned.index, cleaned, color=BLUE, label="После удаления точных дублей", lw=1)
    axes[0].set_title("Влияние очистки транзакций на дневные суммы")
    axes[0].legend()
    axes[1].plot(cleaned.index, cleaned, lw=1)
    flagged = diagnostics.iqr_flag | diagnostics.mad_flag
    axes[1].scatter(diagnostics.index[flagged], diagnostics.units[flagged],
                    color=ORANGE, s=25, label="IQR или |robust Z| > 3.5", zorder=3)
    axes[1].axhline(diagnostics.iqr_upper.iloc[0], color=GREEN, ls="--", label="Верхняя граница IQR")
    axes[1].set_title("Потенциальные выбросы сохранены в исследуемом ряду")
    axes[1].legend()
    for ax in axes:
        date_axis(ax, "Единиц / день")
    return fig


def plot_distribution(series: pd.Series) -> Figure:
    """Compare empirical distributions on original and log1p scales."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, values, title, xlabel in zip(axes, [series, np.log1p(series)],
                                        ["Распределение дневных продаж", "Распределение после log1p"],
                                        ["Единиц / день", "log(1 + единиц / день)"]):
        ax.hist(values, bins=30, color=BLUE, alpha=.85, edgecolor="white")
        ax.axvline(values.mean(), color=ORANGE, label="Среднее", ls="--")
        ax.axvline(values.median(), color=GREEN, label="Медиана", ls=":")
        ax.set(title=title, xlabel=xlabel, ylabel="Число дней")
        ax.legend()
    return fig


def plot_rolling(series: pd.Series, window: int = 28) -> Figure:
    """Use trailing full windows; display no invented estimates at the start."""
    if window < 2 or window > len(series):
        raise ValueError("window must be between 2 and the series length")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained", sharex=True)
    rolling = series.rolling(window, min_periods=window)
    axes[0].plot(series.index, series, alpha=.4, lw=.8, label="Исходный ряд")
    axes[0].plot(series.index, rolling.mean(), lw=2, color=ORANGE, label=f"Среднее за {window} дней")
    axes[0].set_title("Динамика уровня: скользящее среднее")
    axes[0].legend()
    axes[1].plot(series.index, rolling.std(ddof=1), color=GREEN)
    axes[1].set_title(f"Изменение разброса: стандартное отклонение за {window} дней")
    axes[2].plot(series.index, np.log1p(series).rolling(window, min_periods=window).std(), color=GREEN)
    axes[2].set_title(f"Стандартное отклонение после log1p за {window} дней")
    for ax, unit in zip(axes, ["Единиц / день", "Единиц / день", "log(1 + единиц / день)"]):
        date_axis(ax, unit)
    return fig


def plot_calendar(series: pd.Series) -> Figure:
    """Show weekday level and distribution without assuming iid transactions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    grouped = [series[series.index.dayofweek == d].to_numpy() for d in range(7)]
    axes[0].boxplot(grouped, tick_labels=WEEKDAYS, flierprops={"markersize": 3})
    axes[0].set(title="Продажи по дням недели: медиана, IQR, крайние значения",
                xlabel="День недели", ylabel="Единиц / день")
    axes[1].boxplot([np.log1p(x) for x in grouped], tick_labels=WEEKDAYS,
                    flierprops={"markersize": 3})
    axes[1].set(title="Тот же календарный профиль на шкале log1p",
                xlabel="День недели", ylabel="log(1 + единиц / день)")
    return fig


def plot_components(components: pd.DataFrame, scale: str) -> Figure:
    """Plot all four additive STL components with consistent time coordinates."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), layout="constrained", sharex=True)
    labels = {"observed": "Наблюдения", "trend": "Плавный уровень (trend)",
              "seasonal": "Недельная компонента (seasonal)", "residual": "Остаток (residual)"}
    for ax, col in zip(axes, labels):
        ax.plot(components.index, components[col], lw=1, color=GREEN if col == "seasonal" else BLUE)
        ax.set_title(f"{labels[col]} · robust STL, период 7 дней")
        date_axis(ax, scale)
        if col in {"seasonal", "residual"}:
            ax.axhline(0, color=GRAY, lw=.7)
    return fig


def plot_correlations(variants: dict[str, pd.Series], *, pacf: bool = False,
                       max_lags: int = 35) -> Figure:
    """ACF/PACF bands are approximate pointwise 95% bands, not seasonal tests."""
    n = len(variants)
    fig, axes = plt.subplots(n, 2 if pacf else 1, figsize=(12, 3.2 * n),
                             layout="constrained", squeeze=False)
    for i, (name, series) in enumerate(variants.items()):
        lags = min(max_lags, len(series) // 2 - 1)
        plot_acf(series, ax=axes[i, 0], lags=lags, zero=False, alpha=.05, title=f"ACF · {name}")
        if pacf:
            plot_pacf(series, ax=axes[i, 1], lags=lags, zero=False, method="ywm",
                       alpha=.05, title=f"PACF · {name}")
        for ax in axes[i]:
            ax.set(xlabel="Лаг, дней", ylabel="Корреляция")
            ax.set_ylim(-1.05, 1.05)
            for lag in range(7, lags + 1, 7):
                ax.axvline(lag, color=GRAY, lw=.6, alpha=.4)
    return fig


def plot_transforms(variants: dict[str, pd.Series]) -> Figure:
    """Compare transformed time series and 28-day rolling means."""
    fig, axes = plt.subplots(len(variants), 1, figsize=(12, 3 * len(variants)),
                             layout="constrained", squeeze=False)
    for ax, (name, series) in zip(axes.flat, variants.items()):
        ax.plot(series.index, series, lw=.8, label="Наблюдения")
        ax.plot(series.index, series.rolling(28, min_periods=28).mean(),
                color=ORANGE, lw=1.7, label="Среднее за 28 дней")
        ax.axhline(0, color=GRAY, lw=.7)
        ax.set_title(name)
        date_axis(ax, "Логарифмическая шкала")
        ax.legend(loc="upper left")
    return fig
