"""Shared plotting style and chart helpers.

Colors follow a fixed entity->hue assignment (validated for CVD safety):
each strategy keeps its color in every figure. Low-contrast slots (aqua,
yellow) are relieved by direct end-of-line labels plus the metric tables
shown next to every chart.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, metrics

# Surface / ink roles
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed strategy colors — color follows the entity across every figure
STRATEGY_COLORS = {
    "Markowitz Max-Sharpe": "#2a78d6",
    "Michaud Resampled": "#1baf7a",
    "Black-Litterman": "#4a3aa7",
    "Min-Variance": "#e87ba4",
    "60/40": "#eda100",
    "1/N": "#eb6834",
    "SPY (index)": MUTED,   # recessive dashed reference line, not a series
}
INDEX_NAME = "SPY (index)"


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def _ramp(dark: str, light: str, n: int) -> list[str]:
    """n hex steps interpolated dark -> light (one-hue ordinal ramp)."""
    c0, c1 = np.array(mpl.colors.to_rgb(dark)), np.array(mpl.colors.to_rgb(light))
    return [mpl.colors.to_hex(c0 + (c1 - c0) * i / max(n - 1, 1)) for i in range(n)]


def asset_colors(tickers) -> dict[str, str]:
    """Shades of one hue per macro bucket: equity=blue, bond=green, alt=warm."""
    ramps = {"equity": ("#0d366b", "#86b6ef"), "bond": ("#0a5c40", "#8fdec2"),
             "alternative": ("#8a5200", "#f3c96e")}
    colors = {}
    for bucket, (dark, light) in ramps.items():
        members = [t for t in tickers if config.UNIVERSE[t][0] == bucket]
        for t, c in zip(members, _ramp(dark, light, len(members))):
            colors[t] = c
    return colors


def _end_label(ax, series: pd.Series, name: str, color: str):
    ax.annotate(f" {name}", xy=(series.index[-1], series.iloc[-1]),
                color=color if color != MUTED else INK_2,
                fontsize=9, fontweight="bold", va="center")


def plot_investment_curves(returns_dict: dict[str, pd.Series], title: str,
                           initial: float = 100.0, logscale: bool = True):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, r in returns_dict.items():
        color = STRATEGY_COLORS.get(name, MUTED)
        ls = "--" if name == INDEX_NAME else "-"
        curve = metrics.nav(r, initial)
        ax.plot(curve.index, curve.values, color=color, lw=2 if ls == "-" else 1.6,
                ls=ls, label=name, solid_capstyle="round")
        _end_label(ax, curve, name, color)
    if logscale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_title(title, loc="left")
    ax.set_ylabel(f"Growth of ${initial:.0f} (log scale)" if logscale else f"Growth of ${initial:.0f}")
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    ax.margins(x=0.10)
    fig.tight_layout()
    return fig, ax


def plot_drawdowns(returns_dict: dict[str, pd.Series], title: str = "Drawdown"):
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, r in returns_dict.items():
        color = STRATEGY_COLORS.get(name, MUTED)
        dd = metrics.drawdown_series(r)
        ax.plot(dd.index, dd.values, color=color, lw=1.6,
                ls="--" if name == INDEX_NAME else "-", label=name)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", ncols=3, fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_weights(weights: pd.DataFrame, title: str):
    """Stacked area of portfolio weights over time, colored by bucket shades."""
    colors = asset_colors(weights.columns)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(weights.index, weights.T.values,
                 colors=[colors[t] for t in weights.columns],
                 labels=weights.columns, edgecolor=SURFACE, linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title(title, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncols=1)
    ax.margins(x=0)
    fig.tight_layout()
    return fig, ax


def plot_rolling_sharpe(returns_dict: dict[str, pd.Series], rf: pd.Series,
                        window: int = 36, title: str = "Rolling Sharpe (36m)"):
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, r in returns_dict.items():
        color = STRATEGY_COLORS.get(name, MUTED)
        excess = r - rf.reindex(r.index).fillna(0.0)
        roll = excess.rolling(window).mean() / excess.rolling(window).std() * np.sqrt(12)
        ax.plot(roll.index, roll.values, color=color, lw=1.6,
                ls="--" if name == INDEX_NAME else "-", label=name)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    fig.tight_layout()
    return fig, ax


def savefig(fig, name: str):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
