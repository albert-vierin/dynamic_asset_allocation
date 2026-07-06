"""Performance and risk metrics on monthly return series."""

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def nav(returns: pd.Series, initial: float = 100.0) -> pd.Series:
    return initial * (1 + returns).cumprod()


def cagr(returns: pd.Series) -> float:
    n_years = len(returns) / MONTHS_PER_YEAR
    return (1 + returns).prod() ** (1 / n_years) - 1


def ann_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(MONTHS_PER_YEAR)


def sharpe(returns: pd.Series, rf: pd.Series) -> float:
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    if excess.std() == 0:
        return np.nan
    return excess.mean() / excess.std() * np.sqrt(MONTHS_PER_YEAR)


def sortino(returns: pd.Series, rf: pd.Series) -> float:
    excess = returns - rf.reindex(returns.index).fillna(0.0)
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return excess.mean() / downside * np.sqrt(MONTHS_PER_YEAR)


def drawdown_series(returns: pd.Series) -> pd.Series:
    curve = (1 + returns).cumprod()
    return curve / curve.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    return drawdown_series(returns).min()


def calmar(returns: pd.Series) -> float:
    mdd = abs(max_drawdown(returns))
    return cagr(returns) / mdd if mdd > 0 else np.nan


def recovery_months(returns: pd.Series) -> float:
    """Months from the trough of the max drawdown back to the previous peak."""
    dd = drawdown_series(returns)
    trough = dd.idxmin()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    if recovered.empty:
        return np.nan
    return (recovered.index[0].to_period("M") - trough.to_period("M")).n


def summary_table(returns_dict: dict[str, pd.Series], rf: pd.Series,
                  turnover_dict: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per strategy with the full metric set (annualised)."""
    rows = {}
    for name, r in returns_dict.items():
        rows[name] = {
            "CAGR": cagr(r),
            "Ann. vol": ann_vol(r),
            "Sharpe": sharpe(r, rf),
            "Sortino": sortino(r, rf),
            "Max DD": max_drawdown(r),
            "Calmar": calmar(r),
            "Recovery (m)": recovery_months(r),
        }
        if turnover_dict and name in turnover_dict:
            rows[name]["Ann. turnover"] = turnover_dict[name]
    return pd.DataFrame(rows).T


def regime_table(returns_dict: dict[str, pd.Series], regimes: dict, rf: pd.Series,
                 metric: str = "total") -> pd.DataFrame:
    """Strategies x regimes table. metric: 'total' (cum return), 'sharpe' or 'maxdd'."""
    out = {}
    for regime, (start, end) in regimes.items():
        col = {}
        for name, r in returns_dict.items():
            sub = r.loc[start:end]
            if sub.empty:
                col[name] = np.nan
            elif metric == "total":
                col[name] = (1 + sub).prod() - 1
            elif metric == "sharpe":
                col[name] = sharpe(sub, rf)
            elif metric == "maxdd":
                col[name] = max_drawdown(sub)
        out[regime] = col
    return pd.DataFrame(out)
