import yfinance as yf
import pandas as pd
import logging
import os
from datetime import datetime

# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/extract.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

def log_and_print(msg):
    print(msg)
    log.info(msg)

# ── Constants ──────────────────────────────────────────────────────────────────
START = "2008-01-01"
END   = "2025-12-31"

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "NESTLEIND.NS",
    "JSWSTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "TATACONSUM.NS", "DIVISLAB.NS",
    "COALINDIA.NS", "EICHERMOT.NS", "BPCL.NS", "APOLLOHOSP.NS", "BRITANNIA.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "MM.NS", "TATASTEEL.NS", "UPL.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "SHREECEM.NS", "ADANIPORTS.NS"
]

MACRO_TICKERS = {
    "nifty_index" : "^NSEI",
    "india_vix"   : "^INDIAVIX",
    "inr_usd"     : "USDINR=X",
    "gold"        : "GC=F",
    "crude"       : "CL=F",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def validate_download(df, name):
    """Run basic checks on every downloaded dataframe and log results."""
    if df is None or df.empty:
        log_and_print(f"  FAIL {name} — empty dataframe returned")
        return False

    row_count  = len(df)
    date_min   = df.index.min()
    date_max   = df.index.max()
    null_count = df.isnull().sum().sum()
    null_pct   = round(null_count / (df.shape[0] * df.shape[1]) * 100, 2)

    log_and_print(f"  {name}")
    log_and_print(f"    rows      : {row_count}")
    log_and_print(f"    date min  : {date_min.date()}")
    log_and_print(f"    date max  : {date_max.date()}")
    log_and_print(f"    nulls     : {null_count} ({null_pct}%)")

    if row_count < 100:
        log_and_print(f"    WARNING   : very few rows — check ticker or date range")
    if null_pct > 10:
        log_and_print(f"    WARNING   : high null percentage — may need attention")

    return True


# ── D1: NIFTY 50 Stock OHLCV ──────────────────────────────────────────────────
def download_nifty50_ohlcv():
    log_and_print("\n=== D1: Downloading NIFTY 50 OHLCV ===")
    log_and_print(f"Tickers: {len(NIFTY50_TICKERS)}")
    log_and_print(f"Period : {START} to {END}")

    df = yf.download(
        NIFTY50_TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=True
    )

    if df.empty:
        log_and_print("CRITICAL: D1 download returned empty. Check internet and tickers.")
        return

    # Per-ticker missing data check
    log_and_print("\n  Per-ticker missing % (Close column):")
    failed_tickers = []
    if isinstance(df.columns, pd.MultiIndex):
        close_df = df["Close"]
    else:
        close_df = df[["Close"]]

    for ticker in NIFTY50_TICKERS:
        if ticker in close_df.columns:
            missing = close_df[ticker].isnull().sum()
            pct = round(missing / len(close_df) * 100, 2)
            if pct > 5:
                log_and_print(f"    WARNING {ticker}: {pct}% missing")
                failed_tickers.append(ticker)
        else:
            log_and_print(f"    MISSING TICKER: {ticker} not in download result")
            failed_tickers.append(ticker)

    if failed_tickers:
        log_and_print(f"\n  Tickers needing attention: {failed_tickers}")
    else:
        log_and_print("  All 50 tickers present and under 5% missing")

    validate_download(df, "NIFTY50_OHLCV")
    df.to_parquet("data/raw/nifty50_ohlcv.parquet")
    log_and_print("  Saved: data/raw/nifty50_ohlcv.parquet")


# ── D2-D6: Macro Price Tickers ────────────────────────────────────────────────
def download_macro_tickers():
    log_and_print("\n=== D2-D6: Downloading Macro Price Tickers ===")

    for name, ticker in MACRO_TICKERS.items():
        log_and_print(f"\n  Fetching {name} ({ticker})...")

        df = yf.download(
            ticker,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False
        )

        ok = validate_download(df, name)

        if ok:
            close_series = df["Close"].squeeze()
            # Special check for India VIX — starts March 2008
            if name == "india_vix":
                first_valid = close_series.first_valid_index()
                log_and_print(f"    VIX first valid date: {first_valid.date()}")
                log_and_print(f"    Rows before VIX starts (will be NaN): {(df.index < first_valid).sum()}")

            # Special check for crude — April 2020 negative price event
            if name == "crude":
                neg_rows = df[close_series < 0]
                if len(neg_rows) > 0:
                    log_and_print(f"    NOTE: {len(neg_rows)} days with negative crude price (April 2020 event — real, keep it)")

            # Check INR/USD range sanity
            if name == "inr_usd":
                min_val = float(close_series.min())
                max_val = float(close_series.max())
                log_and_print(f"    INR/USD range: {round(min_val, 2)} to {round(max_val, 2)}")
                if min_val < 35 or max_val > 90:
                    log_and_print(f"    WARNING: INR/USD outside expected range 35-90 — check for data errors")

            path = f"data/raw/{name}.parquet"
            df.to_parquet(path)
            log_and_print(f"    Saved: {path}")
        else:
            log_and_print(f"    SKIPPED saving {name} due to empty data")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_and_print(f"\n{'='*60}")
    log_and_print(f"MIDAS-NSE Price Fetcher started: {datetime.now()}")
    log_and_print(f"{'='*60}")

    download_nifty50_ohlcv()
    download_macro_tickers()

    log_and_print(f"\n{'='*60}")
    log_and_print("Price fetcher complete. Check logs/extract.log for full details.")
    log_and_print(f"{'='*60}")