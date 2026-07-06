"""Central configuration for the dynamic asset allocation project.

Every subjective modelling choice required by the assignment (estimation
window, rebalancing frequency, constraints, resampling paths, BL calibration)
lives here so that notebooks and modules stay consistent and each choice is
documented in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# ETF universe
# ---------------------------------------------------------------------------
# 16 US-listed ETFs, all with inception <= mid-2007 so that the sample covers
# >= 18 years and three major crises (GFC 2008, COVID 2020, inflation 2022).
# The universe is chosen to answer the secondary research question: it spans
# asset classes with heterogeneous crisis behaviour (flight-to-quality,
# inflation hedges, credit, real assets).
UNIVERSE = {
    # ticker: (asset class bucket, description)
    "SPY": ("equity", "US large-cap equity (S&P 500)"),
    "IWM": ("equity", "US small-cap equity (Russell 2000)"),
    "QQQ": ("equity", "US growth equity (Nasdaq 100)"),
    "EFA": ("equity", "Developed markets ex-US equity (MSCI EAFE)"),
    "EEM": ("equity", "Emerging markets equity (MSCI EM)"),
    "SHY": ("bond", "US Treasuries 1-3y (cash-like)"),
    "IEF": ("bond", "US Treasuries 7-10y"),
    "TLT": ("bond", "US Treasuries 20y+"),
    "TIP": ("bond", "US inflation-linked Treasuries (TIPS)"),
    "LQD": ("bond", "US investment-grade corporate credit"),
    "HYG": ("bond", "US high-yield corporate credit"),
    "AGG": ("bond", "US aggregate bond market"),
    "GLD": ("alternative", "Gold bullion"),
    "DBC": ("alternative", "Broad commodities"),
    "VNQ": ("alternative", "US REITs"),
    "BIL": ("alternative", "1-3m T-bills (cash proxy)"),
}
TICKERS = list(UNIVERSE)
BUCKETS = sorted({bucket for bucket, _ in UNIVERSE.values()})

# Assets highlighted in the crisis analysis (notebook 04)
CRISIS_HEDGES = ["TLT", "GLD", "TIP", "DBC"]

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
START = "2007-06-01"   # constrained by the youngest ETF inceptions (DBC/BIL: 2006/2007)
END = "2025-12-31"
FREQ = "ME"            # month-end observations

# Risk-free rate: 3-month T-bill from FRED, converted to a monthly series
RF_FRED_SERIES = "TB3MS"

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
ESTIMATION_WINDOW = 60      # months of history used at each rebalance (5y)
ESTIMATION_WINDOW_ALT = 36  # sensitivity check
REBALANCE_FREQ = 1          # months between rebalances (1 = monthly)
REBALANCE_FREQ_ALT = 3      # sensitivity check (quarterly)
OOS_START = "2012-06-30"    # first rebalance date (after the 60m burn-in)
TRANSACTION_COST_BPS = 10   # one-way cost on turnover, robustness scenario

# ---------------------------------------------------------------------------
# Optimisation constraints (long-only, no leverage)
# ---------------------------------------------------------------------------
WEIGHT_CAP = 0.25           # max weight per single ETF
BUCKET_CAP = 0.60           # max total weight per macro bucket
WEIGHT_FLOOR = 0.0

# ---------------------------------------------------------------------------
# Michaud resampling
# ---------------------------------------------------------------------------
N_RESAMPLES = 200           # Monte Carlo paths M (convergence check in nb 02:
                            # weights stable well below Michaud's 500)
RESAMPLE_SEED = 42          # reproducibility

# ---------------------------------------------------------------------------
# Black-Litterman calibration
# ---------------------------------------------------------------------------
BL_TAU = 0.05               # literature value (He-Litterman); sensitivity in nb 02
MOMENTUM_LOOKBACK = 12      # months, 12-1 momentum signal for the views
MOMENTUM_SKIP = 1           # skip most recent month (short-term reversal)
N_VIEW_ASSETS = 3           # top-N vs bottom-N momentum relative view
# Proxy market-cap weights for reverse optimisation (declared, approximate
# relative AUM of the ETFs / strategic multi-asset market portfolio).
MARKET_WEIGHTS = {
    "SPY": 0.28, "IWM": 0.04, "QQQ": 0.10, "EFA": 0.08, "EEM": 0.05,
    "SHY": 0.04, "IEF": 0.05, "TLT": 0.04, "TIP": 0.03, "LQD": 0.05,
    "HYG": 0.03, "AGG": 0.10, "GLD": 0.05, "DBC": 0.02, "VNQ": 0.03,
    "BIL": 0.01,
}

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
BENCHMARK_6040 = {"SPY": 0.60, "AGG": 0.40}
MARKET_INDEX = "SPY"        # buy & hold reference index

# ---------------------------------------------------------------------------
# Crisis regimes (defined ex-ante from well-known market events; month-ends)
# ---------------------------------------------------------------------------
REGIMES = {
    "Taper/China sell-off": ("2015-06-30", "2016-02-29"),
    "Q4 2018 correction": ("2018-09-30", "2018-12-31"),
    "COVID crash": ("2020-01-31", "2020-03-31"),
    "COVID recovery": ("2020-04-30", "2020-12-31"),
    "Inflation shock 2022": ("2022-01-31", "2022-10-31"),
}
