"""Data download, cleaning and preparation.

Pipeline: daily adjusted closes from Yahoo Finance -> quality-control report
-> month-end prices -> simple/log returns. The risk-free rate comes from FRED
(3-month T-bill, TB3MS). All artefacts are cached under data/.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr

from . import config


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_prices(force: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices for the whole universe (cached to parquet)."""
    cache = config.DATA_RAW / "prices_daily_raw.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    raw = yf.download(
        config.TICKERS,
        start=config.START,
        end=config.END,
        auto_adjust=True,   # adjusted for splits and dividends -> total return
        progress=False,
    )["Close"]
    raw = raw[config.TICKERS]
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache)
    return raw


def download_risk_free(force: bool = False) -> pd.Series:
    """Monthly risk-free rate from FRED (TB3MS, annualised %) -> monthly decimal."""
    cache = config.DATA_RAW / "risk_free_monthly.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)["rf"]
    tb3 = pdr.DataReader(config.RF_FRED_SERIES, "fred", config.START, config.END)
    rf = ((1 + tb3[config.RF_FRED_SERIES] / 100) ** (1 / 12) - 1).rename("rf")
    rf.index = rf.index + pd.offsets.MonthEnd(0)  # align to month-end stamps
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    rf.to_frame().to_parquet(cache)
    return rf


# ---------------------------------------------------------------------------
# Quality control and cleaning
# ---------------------------------------------------------------------------

def quality_report(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker data-quality summary used to document the cleaning step."""
    rets = daily.pct_change()
    report = pd.DataFrame({
        "first_obs": daily.apply(lambda s: s.first_valid_index()),
        "last_obs": daily.apply(lambda s: s.last_valid_index()),
        "n_obs": daily.notna().sum(),
        "n_missing_inside": [
            daily[c].loc[daily[c].first_valid_index():].isna().sum()
            for c in daily.columns
        ],
        "n_zero_price": (daily <= 0).sum(),
        "max_abs_daily_ret": rets.abs().max(),
        "n_daily_ret_gt_20pct": (rets.abs() > 0.20).sum(),
    })
    report.index.name = "ticker"
    return report


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end prices; forward-fill inside the month only (holiday gaps)."""
    filled = daily.ffill(limit=5)
    monthly = filled.resample(config.FREQ).last()
    return monthly.dropna(how="any")  # start when all 16 ETFs are alive


def compute_returns(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    simple = monthly.pct_change().dropna(how="any")
    log = np.log(monthly).diff().dropna(how="any")
    return simple, log


# ---------------------------------------------------------------------------
# One-shot loader used by every notebook
# ---------------------------------------------------------------------------

def load_dataset(force: bool = False):
    """Returns (monthly prices, simple returns, monthly rf) building caches as needed."""
    prices_cache = config.DATA_PROCESSED / "prices_monthly.parquet"
    returns_cache = config.DATA_PROCESSED / "returns_monthly.parquet"
    if prices_cache.exists() and returns_cache.exists() and not force:
        monthly = pd.read_parquet(prices_cache)
        simple = pd.read_parquet(returns_cache)
    else:
        daily = download_prices(force=force)
        monthly = to_monthly(daily)
        simple, _ = compute_returns(monthly)
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        monthly.to_parquet(prices_cache)
        simple.to_parquet(returns_cache)
    rf = download_risk_free(force=force).reindex(simple.index).ffill()
    return monthly, simple, rf
