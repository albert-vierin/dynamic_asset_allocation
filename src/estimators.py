"""Input estimators: expected returns and covariance matrices.

All estimators take a window of *monthly simple returns* (DataFrame) and
return annualised quantities, the convention used by PyPortfolioOpt.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

MONTHS_PER_YEAR = 12


def mean_historical(window: pd.DataFrame) -> pd.Series:
    """Annualised arithmetic mean of monthly returns."""
    return window.mean() * MONTHS_PER_YEAR


def cov_sample(window: pd.DataFrame) -> pd.DataFrame:
    """Annualised sample covariance."""
    return window.cov() * MONTHS_PER_YEAR


def cov_ledoit_wolf(window: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf shrinkage covariance (constant-variance target)."""
    lw = LedoitWolf().fit(window.values)
    cov = lw.covariance_ * MONTHS_PER_YEAR
    return pd.DataFrame(cov, index=window.columns, columns=window.columns)


def momentum_12_1(prices: pd.Series | pd.DataFrame, lookback: int = 12, skip: int = 1):
    """Classic 12-1 momentum: total return from t-12 to t-1 (monthly prices)."""
    return prices.shift(skip) / prices.shift(lookback) - 1
