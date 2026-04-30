import yfinance as yf
import pandas as pd
import logging
import os

logging.basicConfig(
    filename="logs/extract.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

def log_and_print(msg):
    print(msg)
    log.info(msg)

START = "2008-01-01"
END   = "2025-12-31"

# Correct tickers to retry
FAILED_TICKERS = {
    "M&M.NS"        : "MM.NS",       # correct → was wrong
    "TATAMOTORS.NS" : "TATAMOTORS.NS" # correct ticker, retry individually
}

log_and_print("\n=== Fixing Failed Tickers ===")

# Load existing parquet to patch into it
existing = pd.read_parquet("data/raw/nifty50_ohlcv.parquet")
log_and_print(f"Existing shape before fix: {existing.shape}")

for correct_ticker, old_ticker in FAILED_TICKERS.items():
    log_and_print(f"\nFetching {correct_ticker} individually...")

    df = yf.download(
        correct_ticker,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        log_and_print(f"  FAIL: {correct_ticker} still returned empty")
        continue

    log_and_print(f"  Rows fetched : {len(df)}")
    log_and_print(f"  Date range   : {df.index.min().date()} to {df.index.max().date()}")
    log_and_print(f"  Nulls        : {df.isnull().sum().sum()}")

    # Patch each OHLCV column into the existing dataframe
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            existing[(col, correct_ticker)] = df[col]
            log_and_print(f"  Patched column: ({col}, {correct_ticker})")

log_and_print(f"\nShape after fix: {existing.shape}")

# Save updated parquet
existing.to_parquet("data/raw/nifty50_ohlcv.parquet")
log_and_print("Saved: data/raw/nifty50_ohlcv.parquet (updated)")

# Final missing check on the two fixed tickers
log_and_print("\n--- Post-fix missing check ---")
close_df = existing["Close"]
for ticker in ["M&M.NS", "TATAMOTORS.NS"]:
    if ticker in close_df.columns:
        missing = close_df[ticker].isnull().sum()
        pct = round(missing / len(close_df) * 100, 2)
        log_and_print(f"  {ticker}: {pct}% missing")
    else:
        log_and_print(f"  {ticker}: still not found in columns")