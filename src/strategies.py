"""Strategy functions with the signature expected by backtest.run_backtest:

    fn(hist_returns, hist_prices, rf_annual) -> pd.Series of weights

hist_returns is the estimation window (monthly simple returns up to the
rebalance date), hist_prices the full price history up to the same date.
"""

from . import config, estimators, optimizers
from .backtest import equal_weight_strategy, fixed_weight_strategy  # noqa: F401


def markowitz_max_sharpe(hist_ret, hist_prc, rf_annual):
    mu = estimators.mean_historical(hist_ret)
    cov = estimators.cov_ledoit_wolf(hist_ret)
    return optimizers.max_sharpe(mu, cov, rf_annual)


def markowitz_min_variance(hist_ret, hist_prc, rf_annual):
    mu = estimators.mean_historical(hist_ret)
    cov = estimators.cov_ledoit_wolf(hist_ret)
    return optimizers.min_variance(mu, cov)


def michaud(hist_ret, hist_prc, rf_annual):
    mu = estimators.mean_historical(hist_ret)
    cov = estimators.cov_ledoit_wolf(hist_ret)
    return optimizers.michaud_resampled(mu, cov, rf_annual, n_obs=len(hist_ret))


def black_litterman(hist_ret, hist_prc, rf_annual):
    cov = estimators.cov_ledoit_wolf(hist_ret)
    w, _ = optimizers.black_litterman_weights(
        hist_prc, cov, rf_annual, market_prices=hist_prc[config.MARKET_INDEX]
    )
    return w


benchmark_6040 = fixed_weight_strategy(config.BENCHMARK_6040)

STRATEGIES = {
    "Markowitz Max-Sharpe": markowitz_max_sharpe,
    "Michaud Resampled": michaud,
    "Black-Litterman": black_litterman,
    "60/40": benchmark_6040,
    "1/N": equal_weight_strategy,
}
