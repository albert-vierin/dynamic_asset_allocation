"""Portfolio optimisers: constrained Markowitz, Michaud resampling, Black-Litterman.

All functions take annualised (mu, cov) inputs and return a pd.Series of
weights satisfying the common constraint set: long-only, no leverage,
single-asset cap, macro-bucket cap.
"""

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, black_litterman
from pypfopt.black_litterman import BlackLittermanModel

from . import config, estimators


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def _bucket_maps(tickers):
    mapper = {t: config.UNIVERSE[t][0] for t in tickers}
    upper = {b: config.BUCKET_CAP for b in config.BUCKETS}
    lower = {b: 0.0 for b in config.BUCKETS}
    return mapper, lower, upper


def _frontier(mu: pd.Series, cov: pd.DataFrame) -> EfficientFrontier:
    ef = EfficientFrontier(mu, cov, weight_bounds=(config.WEIGHT_FLOOR, config.WEIGHT_CAP))
    mapper, lower, upper = _bucket_maps(mu.index)
    ef.add_sector_constraints(mapper, lower, upper)
    return ef


def _clean(weights: dict, tickers) -> pd.Series:
    w = pd.Series(weights).reindex(tickers).fillna(0.0)
    w[w < 1e-5] = 0.0
    return w / w.sum()


# ---------------------------------------------------------------------------
# (i) Classical Markowitz
# ---------------------------------------------------------------------------

def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float) -> pd.Series:
    """Constrained tangency portfolio; falls back to min-variance when no
    asset beats the risk-free rate (max-Sharpe is then undefined)."""
    try:
        ef = _frontier(mu, cov)
        ef.max_sharpe(risk_free_rate=rf)
        return _clean(ef.clean_weights(), mu.index)
    except Exception:
        return min_variance(mu, cov)


def min_variance(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    ef = _frontier(mu, cov)
    ef.min_volatility()
    return _clean(ef.clean_weights(), mu.index)


# ---------------------------------------------------------------------------
# (ii) Michaud resampled portfolio
# ---------------------------------------------------------------------------

def michaud_resampled(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float,
    n_obs: int = config.ESTIMATION_WINDOW,
    n_resamples: int = config.N_RESAMPLES,
    seed: int = config.RESAMPLE_SEED,
    base_optimizer=max_sharpe,
) -> pd.Series:
    """Michaud & Michaud (2008) resampling.

    For each Monte Carlo path: simulate n_obs monthly returns from
    N(mu/12, cov/12), re-estimate (mu_m, cov_m), solve the same constrained
    problem, then average the optimal weights across paths.
    """
    rng = np.random.default_rng(seed)
    tickers = mu.index
    weights = []
    for _ in range(n_resamples):
        sim = rng.multivariate_normal(mu.values / 12, cov.values / 12, size=n_obs)
        sim = pd.DataFrame(sim, columns=tickers)
        mu_m = estimators.mean_historical(sim)
        cov_m = estimators.cov_ledoit_wolf(sim)
        weights.append(base_optimizer(mu_m, cov_m, rf) if base_optimizer is max_sharpe
                       else base_optimizer(mu_m, cov_m))
    avg = pd.concat(weights, axis=1).mean(axis=1)
    return avg / avg.sum()


# ---------------------------------------------------------------------------
# (iii) Black-Litterman with momentum views (Idzorek confidence)
# ---------------------------------------------------------------------------

def momentum_view(prices_window: pd.DataFrame):
    """Build one relative view from the 12-1 momentum signal.

    View: the equal-weighted basket of the top-N momentum assets outperforms
    the bottom-N basket. Q assumes half of the trailing 12m spread persists
    over the next year (conservative persistence assumption). Idzorek
    confidence scales linearly with the size of the spread.
    """
    mom = estimators.momentum_12_1(
        prices_window, config.MOMENTUM_LOOKBACK, config.MOMENTUM_SKIP
    ).iloc[-1].dropna()
    n = config.N_VIEW_ASSETS
    top, bottom = mom.nlargest(n).index, mom.nsmallest(n).index
    spread = mom[top].mean() - mom[bottom].mean()

    P = pd.Series(0.0, index=prices_window.columns)
    P[top], P[bottom] = 1.0 / n, -1.0 / n
    Q = 0.5 * spread
    confidence = float(np.clip(abs(spread) / 0.40, 0.20, 0.80))
    return P, Q, confidence


def black_litterman_weights(
    prices_window: pd.DataFrame,
    cov: pd.DataFrame,
    rf: float,
    market_prices: pd.Series,
    tau: float = config.BL_TAU,
) -> tuple[pd.Series, dict]:
    """BL posterior from equilibrium prior + one momentum view -> max-Sharpe.

    Returns (weights, diagnostics dict with pi, posterior mu, view details).
    """
    tickers = prices_window.columns
    mkt_w = pd.Series(config.MARKET_WEIGHTS).reindex(tickers)
    mkt_w = mkt_w / mkt_w.sum()

    delta = black_litterman.market_implied_risk_aversion(
        market_prices, frequency=12, risk_free_rate=rf)  # monthly prices
    delta = float(np.clip(delta, 1.0, 6.0))  # guard against unstable estimates
    pi = black_litterman.market_implied_prior_returns(mkt_w, delta, cov, risk_free_rate=rf)

    P, Q, conf = momentum_view(prices_window)
    bl = BlackLittermanModel(
        cov,
        pi=pi,
        P=P.values.reshape(1, -1),
        Q=np.array([Q]),
        omega="idzorek",
        view_confidences=np.array([conf]),
        tau=tau,
        risk_aversion=delta,
    )
    mu_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()
    w = max_sharpe(mu_bl, cov_bl, rf)
    diag = {"pi": pi, "mu_bl": mu_bl, "P": P, "Q": Q, "confidence": conf, "delta": delta}
    return w, diag
