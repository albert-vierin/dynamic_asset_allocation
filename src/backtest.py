"""Walk-forward backtesting engine.

At each rebalance date t (a month-end), a strategy sees only data up to and
including month t, produces target weights, and those weights earn the
returns of the following months until the next rebalance. Between rebalances
weights drift with realised returns. Transaction costs are charged on the
units traded at each rebalance.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series                 # net of transaction costs
    gross_returns: pd.Series
    weights: pd.DataFrame              # target weights at each rebalance date
    turnover: pd.Series                # sum |dw| at each rebalance (two-sided)
    diagnostics: dict = field(default_factory=dict)

    @property
    def annual_turnover(self) -> float:
        """Average one-way turnover per year."""
        n_years = len(self.returns) / 12
        return self.turnover.sum() / 2 / n_years


def rebalance_dates(index: pd.DatetimeIndex, oos_start=config.OOS_START,
                    every: int = config.REBALANCE_FREQ) -> pd.DatetimeIndex:
    dates = index[index >= pd.Timestamp(oos_start)]
    return dates[:-1:every]  # last month has no following return to invest in


def run_backtest(
    strategy_fn,
    name: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    rf: pd.Series,
    window: int = config.ESTIMATION_WINDOW,
    every: int = config.REBALANCE_FREQ,
    tc_bps: float = 0.0,
    oos_start=config.OOS_START,
) -> BacktestResult:
    """strategy_fn(hist_returns, hist_prices, rf_annual) -> pd.Series of weights.

    hist_returns / hist_prices contain *only* observations up to the rebalance
    date (the estimation window for returns, the full history for prices, as
    Black-Litterman needs a longer price series for its momentum signal).
    """
    rebal = rebalance_dates(returns.index, oos_start, every)
    port_gross, port_net, dates_out = [], [], []
    weights_log, turnover_log = {}, {}
    w_prev = pd.Series(0.0, index=returns.columns)  # start from cash

    for t in rebal:
        hist_ret = returns.loc[:t].tail(window)
        hist_prc = prices.loc[:t]
        rf_annual = float((1 + rf.loc[t]) ** 12 - 1)

        w_target = strategy_fn(hist_ret, hist_prc, rf_annual)
        w_target = w_target.reindex(returns.columns).fillna(0.0)
        weights_log[t] = w_target
        turnover = float((w_target - w_prev).abs().sum())
        turnover_log[t] = turnover

        # holding period: the `every` months following the rebalance date
        future = returns.loc[returns.index > t].head(every)
        w = w_target.copy()
        for i, (date, r_vec) in enumerate(future.iterrows()):
            r_p = float(w @ r_vec)
            cost = (tc_bps / 1e4) * turnover if i == 0 else 0.0
            port_gross.append(r_p)
            port_net.append(r_p - cost)
            dates_out.append(date)
            w = w * (1 + r_vec) / (1 + r_p)  # drift until next rebalance

        w_prev = w

    idx = pd.DatetimeIndex(dates_out)
    return BacktestResult(
        name=name,
        returns=pd.Series(port_net, index=idx, name=name),
        gross_returns=pd.Series(port_gross, index=idx, name=name),
        weights=pd.DataFrame(weights_log).T,
        turnover=pd.Series(turnover_log),
    )


CACHE_DIR = config.DATA_PROCESSED / "backtests"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower())


def save_result(res: BacktestResult):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slug(res.name)
    pd.DataFrame({"net": res.returns, "gross": res.gross_returns}).to_parquet(
        CACHE_DIR / f"{slug}_returns.parquet")
    res.weights.to_parquet(CACHE_DIR / f"{slug}_weights.parquet")
    res.turnover.to_frame("turnover").to_parquet(CACHE_DIR / f"{slug}_turnover.parquet")


def load_result(name: str) -> BacktestResult | None:
    slug = _slug(name)
    path = CACHE_DIR / f"{slug}_returns.parquet"
    if not path.exists():
        return None
    rets = pd.read_parquet(path)
    return BacktestResult(
        name=name,
        returns=rets["net"].rename(name),
        gross_returns=rets["gross"].rename(name),
        weights=pd.read_parquet(CACHE_DIR / f"{slug}_weights.parquet"),
        turnover=pd.read_parquet(CACHE_DIR / f"{slug}_turnover.parquet")["turnover"],
    )


def run_or_load(strategy_fn, name, returns, prices, rf, force: bool = False,
                **kwargs) -> BacktestResult:
    """Backtests are deterministic (fixed seed), so cache them to parquet:
    the slow Michaud run (rebalances x Monte Carlo paths) executes once."""
    if not force:
        cached = load_result(name)
        if cached is not None:
            return cached
    res = run_backtest(strategy_fn, name, returns, prices, rf, **kwargs)
    save_result(res)
    return res


def fixed_weight_strategy(target: dict):
    """Naive benchmark: same target weights at every rebalance (e.g. 60/40)."""
    def fn(hist_ret, hist_prc, rf_annual):
        w = pd.Series(0.0, index=hist_ret.columns)
        for k, v in target.items():
            w[k] = v
        return w / w.sum()
    return fn


def equal_weight_strategy(hist_ret, hist_prc, rf_annual):
    """1/N across the whole universe."""
    n = hist_ret.shape[1]
    return pd.Series(1.0 / n, index=hist_ret.columns)
