"""Графики для ноутбука и страницы ДЗ на GitHub."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from forecasting import BASELINE

INK = "#152942"
MUTED = "#52667B"
GRID = "#DEE7ED"
TEAL = "#087F82"
GOLD = "#F6BF68"
COLORS = {
    "Наивная": "#B36610", "Сглаживание": "#138378", BASELINE: "#647389",
    "AutoReg": "#4266C3", "ARIMA": "#AC396A", "Бустинг": "#6E47B2",
}
NAMES = {BASELINE: "Исходная AutoReg"}
FREQUENCIES = {"День": "Дни", "Неделя": "Недели", "Месяц": "Месяцы"}
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def heading(fig, title, subtitle, dark=False, left=0.065):
    height = fig.get_figheight()
    fig.text(left, 1 - 0.25 / height, title, va="top", size=21, weight="bold",
             color="#F5FAFE" if dark else INK)
    fig.text(left, 1 - 0.68 / height, subtitle, va="top", size=11,
             color="#C6D7E7" if dark else MUTED)


def tidy(ax):
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, pad=6, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def save_figure(fig, folder, name):
    folder.mkdir(exist_ok=True, parents=True)
    fig.savefig(folder / name, dpi=160, facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)


def sales_history(sales, test_start):
    fig = plt.figure(figsize=(11, 4.7), facecolor=INK)
    heading(fig, "Дневные продажи за год",
            "LUNCH BAG CARS BLUE  ·  товар 20728  ·  декабрь 2010 — декабрь 2011", dark=True)
    ax = fig.add_axes([0.078, 0.245, 0.895, 0.505], facecolor=INK)
    ax.axvspan(test_start, sales.index[-1], color="#7965BB", alpha=0.30, linewidth=0)
    ax.axvline(test_start, color="#C9B7F6", alpha=0.8, ls="--", lw=1)
    ax.vlines(sales.index, 0, sales, color="#71CFCC", linewidth=1.35, alpha=0.86)
    trend = sales.rolling(7, min_periods=7).mean()
    ax.plot(trend, color=GOLD, lw=2.1, zorder=3)
    peak = sales.idxmax()
    ax.annotate(f"{sales.max():.0f} шт.  ·  {peak:%d.%m}",
                (peak, sales.max()), xytext=(-13, 7), textcoords="offset points",
                ha="right", color="white", weight="bold", fontsize=10)
    ax.set_xlim(sales.index.min(), sales.index.max())
    ax.set_ylim(0, sales.max() * 1.15)
    ax.set_ylabel("Продажи, шт.", color="#C6D7E7", labelpad=8)
    ax.set_xlabel("Месяц", color="#C6D7E7", labelpad=5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{MONTHS[mdates.num2date(x).month - 1]} {mdates.num2date(x):%y}"))
    ax.tick_params(colors="#D7E5EE", length=0, pad=6)
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", color="white", alpha=0.13)
    ax.set_axisbelow(True)
    legend = [
        Line2D([], [], color="#71CFCC", lw=3, label="Продажи за день"),
        Line2D([], [], color=GOLD, lw=2.5, label="Среднее за 7 дней"),
        Patch(facecolor="#615788", label="Дневной тест"),
    ]
    fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.062, 0.035),
               ncol=3, frameon=False, labelcolor="#E2EEF8", fontsize=11)
    return fig


def error_matrix(metrics):
    fig = plt.figure(figsize=(11, 4.85), facecolor="white")
    heading(fig, "Ошибки моделей на девяти горизонтах",
            "WAPE, %. Более тёмная ячейка означает большую ошибку. Рамкой выделен минимум в столбце.")
    model_names = list(COLORS)
    groups = [metrics.loc[metrics.frequency == key] for key in FREQUENCIES]
    widths = [g.horizon.nunique() for g in groups]
    grid = fig.add_gridspec(1, 3, width_ratios=widths, left=0.185, right=0.97,
                            bottom=0.16, top=0.70, wspace=0.12)
    cmap = LinearSegmentedColormap.from_list("errors", ["#E8F8F3", "#8EC9C5", "#487E94", "#233F61"])
    norm = Normalize(vmin=0, vmax=np.ceil(metrics.WAPE_pct.max() / 10) * 10)
    for i, (label, group) in enumerate(zip(FREQUENCIES.values(), groups)):
        ax = fig.add_subplot(grid[i])
        mat = group.pivot(index="model", columns="horizon", values="WAPE_pct").loc[model_names]
        ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for col, horizon in enumerate(mat.columns):
            values = mat[horizon].to_numpy()
            for row, value in enumerate(values):
                best = np.isclose(value, values.min())
                if best:
                    ax.add_patch(Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                           fill=False, edgecolor=INK, linewidth=2))
                rgb = np.array(cmap(norm(value))[:3])
                color = "white" if rgb @ [0.2126, 0.7152, 0.0722] < 0.55 else INK
                ax.text(col, row, f"{value:.1f}".replace(".", ","), ha="center", va="center",
                        fontsize=11, color=color, weight="bold" if best else "normal")
        ax.set_title(label, color=INK, fontsize=13, weight="bold", pad=9)
        ax.set_xticks(range(len(mat.columns)), mat.columns)
        ax.set_yticks(range(len(mat)), [NAMES.get(m, m) for m in model_names] if i == 0 else [])
        ax.tick_params(length=0, labelsize=10, pad=7, colors=INK)
        ax.spines[:].set_visible(False)
    fig.text(0.58, 0.035, "Горизонт прогноза, шагов", ha="center", color=MUTED, size=10)
    return fig


def forecast_panels(predictions, label, horizons):
    data = predictions.loc[predictions.frequency == label]
    start = data.cutoff.min()
    example = data.loc[data.cutoff == start]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.65), sharex=True, sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.975, top=0.75, bottom=0.18, wspace=0.22, hspace=0.45)
    heading(fig, f"{FREQUENCIES[label]}. Прогноз каждой модели",
            f"Первый старт теста — {pd.Timestamp(start):%d.%m.%Y}. На оси отмечены горизонты из задания.")
    for ax, (name, color) in zip(axes.flat, COLORS.items()):
        path = example.loc[example.model == name].sort_values("step")
        tidy(ax)
        ax.set_title(NAMES.get(name, name), loc="left", color=color, fontsize=12, weight="bold")
        ax.fill_between(path.step, 0, path.actual, color="#D4E4EB", alpha=0.65)
        marker = "o" if len(path) <= 8 else None
        ax.plot(path.step, path.actual, color=INK, lw=1.45, marker=marker, ms=3)
        ax.plot(path.step, path.forecast, color=color, lw=1.9, ls="--", marker=marker, ms=3)
        ax.set_xticks(horizons)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    peak = example[["actual", "forecast"]].to_numpy().max()
    axes.flat[0].set_ylim(0, max(1, peak * 1.08))
    fig.text(0.022, 0.47, "Продажи, шт.", rotation=90, va="center", color=MUTED)
    fig.text(0.53, 0.074, "Шаг прогноза", ha="center", color=MUTED)
    fig.legend(handles=[
        Line2D([], [], color=INK, label="Факт"),
        Line2D([], [], color=MUTED, ls="--", label="Прогноз модели"),
    ], loc="lower left", bbox_to_anchor=(0.065, 0.004), ncol=2, frameon=False)
    return fig


def horizon_errors(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True, facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.29, top=0.70, wspace=0.17)
    heading(fig, "Как ошибка меняется с горизонтом",
            "Каждая точка учитывает все старты, для которых известен факт на выбранном горизонте.")
    for ax, (key, label) in zip(axes, FREQUENCIES.items()):
        tidy(ax)
        ax.set_title(label, loc="left", weight="bold", fontsize=13, color=INK)
        data = metrics.loc[metrics.frequency == key]
        for name, color in COLORS.items():
            group = data.loc[data.model == name].sort_values("horizon")
            # Одинаковые настройки двух AutoReg на днях дают совпадающие линии.
            ax.plot(group.horizon.astype(str), group.WAPE_pct, marker="o", markersize=4,
                    color=color, lw=1.7, ls=":" if name == BASELINE else "-",
                    label=NAMES.get(name, name))
        ax.margins(x=0.09)
    axes[0].set_ylabel("WAPE, %")
    fig.text(0.53, 0.185, "Горизонт, шагов", ha="center", color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False)
    return fig


def daily_diagnostics(predictions):
    data = predictions.loc[(predictions.frequency == "День") &
                           (predictions.model == "Бустинг") & (predictions.step == 1)].copy()
    data["weekday"] = pd.to_datetime(data.date).dt.dayofweek
    means = data.groupby("weekday")[["actual", "forecast"]].mean().reindex(range(7))
    fig, (bars, scatter) = plt.subplots(1, 2, figsize=(11, 4.95), facecolor="white",
                                        gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.085, right=0.97, top=0.69, bottom=0.22, wspace=0.32)
    heading(fig, "Когда ошибается дневной бустинг",
            f"Прогноз на один день. В тесте {len(data)} дат, каждая учтена один раз.")
    tidy(bars)
    tidy(scatter)
    bars.set_title("Средние продажи по дням недели", loc="left", size=12, weight="bold", color=INK)
    x = np.arange(7)
    bars.bar(x - 0.18, means.actual, 0.36, color=INK, label="Факт")
    bars.bar(x + 0.18, means.forecast, 0.36, color=COLORS["Бустинг"], label="Бустинг")
    bars.set_xticks(x, ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    bars.set(ylabel="Продажи, шт.", xlabel="День недели")
    bars.legend(loc="upper left", bbox_to_anchor=(0, -0.27), frameon=False, ncol=2, borderaxespad=0)
    scatter.set_title("Каждая точка — один день", loc="left", size=12, weight="bold", color=INK)
    limit = max(data.actual.max(), data.forecast.max()) * 1.055
    scatter.plot([0, limit], [0, limit], lw=1.5, ls="--", color=MUTED, label="Точный прогноз")
    scatter.scatter(data.actual, data.forecast, s=26, facecolor=TEAL, edgecolor="white",
                    linewidth=0.4, alpha=0.72)
    peak = data.loc[data.actual.idxmax()]
    scatter.annotate(f"{pd.Timestamp(peak.date):%d.%m}  ·  {peak.actual:.0f} шт.",
                     (peak.actual, peak.forecast), xytext=(-6, 12),
                     textcoords="offset points", ha="right", color=TEAL, fontsize=9)
    scatter.set(xlabel="Фактические продажи, шт.", ylabel="Прогноз, шт.",
                xlim=(-3, limit), ylim=(-3, limit))
    scatter.legend(loc="upper left", frameon=False)
    return fig
