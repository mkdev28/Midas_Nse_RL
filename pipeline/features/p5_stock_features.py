# pipeline/features/p5_stock_features.py
import sys, os
import pandas as pd
import numpy as np
import joblib

sys.path.append(r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse")
from config import PROC, RAW

# ── Load OHLCV ────────────────────────────────────────────────────────
print("Loading OHLCV parquet...")
ohlcv = pd.read_parquet(os.path.join(RAW, "nifty50_ohlcv.parquet"))
ohlcv.index = pd.to_datetime(ohlcv.index)
ohlcv.index.name = "date"

# ── Load trading calendar ─────────────────────────────────────────────
calendar     = pd.read_parquet(os.path.join(PROC, "trading_calendar.parquet"))
calendar["date"] = pd.to_datetime(calendar["date"])
trading_days = calendar["date"].values
print(f"Trading days in calendar : {len(trading_days)}")

# ── Tickers ───────────────────────────────────────────────────────────
close_tickers = [t for t in ohlcv["Close"].columns.tolist() if t != 'MM.NS']
print(f"Tickers with Close data  : {len(close_tickers)}")
print(f"Sample tickers           : {close_tickers[:5]}")

# ── Feature computation ───────────────────────────────────────────────
def compute_stock_features(close, high, low, volume):
    feat = pd.DataFrame(index=close.index)

    # Pre-compute EMAs needed for multiple features
    ema5  = close.ewm(span=5,  adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    # 1 & 2 — EMA ratios (price-scale-free)
    feat["ema_5"]  = close / ema5  - 1
    feat["ema_20"] = close / ema20 - 1

    # 3 — RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.where(loss != 0, other=1e-10)   # Fix 1
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # 4 & 5 — MACD (price-scale-free)
    macd              = ema12 - ema26
    macd_signal       = macd.ewm(span=9, adjust=False).mean()
    feat["macd"]      = macd / close                      # Fix 4
    feat["macd_hist"] = (macd - macd_signal) / close      # Fix 4

    # 6 — ATR 14 (normalised)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    feat["atr_14"] = tr.rolling(14).mean() / close        # Fix 4

    # 7 — Bollinger Band width (normalised)
    sma20           = close.rolling(20).mean()
    std20           = close.rolling(20).std()
    feat["bb_width"] = (4 * std20) / sma20

    # 8 — OBV rate of change (Fix 3 — not cumulative)
    direction       = np.sign(close.diff()).fillna(0)
    obv_raw         = (direction * volume).cumsum()
    feat["obv"]     = obv_raw.pct_change().fillna(0)

    # 9 — MFI 14
    typical_price   = (high + low + close) / 3
    raw_mf          = typical_price * volume
    pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio       = pos_mf / neg_mf.where(neg_mf != 0, other=1e-10)
    feat["mfi_14"]  = 100 - (100 / (1 + mfi_ratio))

    # 10 & 11 — Returns
    feat["daily_return"] = close.pct_change()
    feat["log_return"]   = np.log(close / close.shift(1))

    # 12 — Rolling Sharpe 30 (Fix 2)
    roll_mean = feat["daily_return"].rolling(30).mean()
    roll_std  = feat["daily_return"].rolling(30).std()
    feat["rolling_sharpe_30"] = (
        roll_mean / roll_std.where(roll_std != 0, other=1e-10) * np.sqrt(252)
    )

    return feat


FEATURE_NAMES = [
    "ema_5", "ema_20", "rsi_14", "macd", "macd_hist",
    "atr_14", "bb_width", "obv", "mfi_14",
    "daily_return", "log_return", "rolling_sharpe_30"
]
N_FEATURES = len(FEATURE_NAMES)  # 12

# ── Process all tickers ───────────────────────────────────────────────
all_stock_features = {}
failed_tickers     = []

print(f"\nComputing {N_FEATURES} features for {len(close_tickers)} tickers...")

for i, ticker in enumerate(close_tickers):
    try:
        close  = ohlcv["Close"][ticker].reindex(trading_days)

        volume = (ohlcv["Volume"][ticker].reindex(trading_days)
                  if ticker in ohlcv["Volume"].columns
                  else pd.Series(0, index=trading_days))

        if ticker in ohlcv["High"].columns:
            high = ohlcv["High"][ticker].reindex(trading_days)
            low  = ohlcv["Low"][ticker].reindex(trading_days)
        else:
            high = close
            low  = close

        feat_df = compute_stock_features(close, high, low, volume)
        all_stock_features[ticker] = feat_df

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(close_tickers)} tickers")

    except Exception as e:
        print(f"  FAILED: {ticker} — {e}")
        failed_tickers.append(ticker)

print(f"\nSuccessfully processed : {len(all_stock_features)} tickers")
print(f"Failed                 : {len(failed_tickers)} — {failed_tickers}")

# ── Build 3D array [dates, tickers, features] ─────────────────────────
n_dates    = len(trading_days)
n_tickers  = len(close_tickers)
feat_array = np.full((n_dates, n_tickers, N_FEATURES), np.nan, dtype=np.float32)
ticker_index = []

for j, ticker in enumerate(close_tickers):
    if ticker in all_stock_features:
        df_feat = all_stock_features[ticker].reindex(trading_days)
        df_feat = df_feat.ffill().bfill()
        for k, fname in enumerate(FEATURE_NAMES):
            if fname in df_feat.columns:
                feat_array[:, j, k] = df_feat[fname].values
        ticker_index.append(ticker)

print(f"\nFeature array shape : {feat_array.shape}")

# ── NaN report ────────────────────────────────────────────────────────
nan_count = np.isnan(feat_array).sum()
nan_pct   = round(nan_count / feat_array.size * 100, 3)
print(f"NaN in feature array  : {nan_count} ({nan_pct}%)")
print("  NaN level acceptable (warmup + IPO gaps expected)"
      if nan_pct <= 5 else "  WARNING: High NaN — check missing tickers")

# ── Normalize on train only (Fix 5) ───────────────────────────────────
trading_days_dt = pd.to_datetime(trading_days)
train_mask = trading_days_dt <  "2021-01-01"
val_mask   = (trading_days_dt >= "2021-01-01") & (trading_days_dt < "2023-01-01")
test_mask  = trading_days_dt >= "2023-01-01"

train_data = feat_array[train_mask]
means = np.nanmean(train_data, axis=(0, 1), keepdims=True)   # (1, 1, 12)
stds  = np.nanstd(train_data,  axis=(0, 1), keepdims=True)
stds[stds == 0] = 1

feat_array = ((feat_array - means) / stds).astype(np.float32)
feat_array = np.nan_to_num(feat_array, nan=0.0)              # fill residual NaN with 0

print(f"\nNormalized — train mean : {feat_array[train_mask].mean():.4f}")
print(f"Normalized — train std  : {feat_array[train_mask].std():.4f}")

# ── Save ──────────────────────────────────────────────────────────────
np.save(os.path.join(PROC, "stock_features.npy"),       feat_array)
np.save(os.path.join(PROC, "stock_features_means.npy"), means.astype(np.float32))
np.save(os.path.join(PROC, "stock_features_stds.npy"),  stds.astype(np.float32))

np.save(os.path.join(PROC, "X_train_technical.npy"), feat_array[train_mask])
np.save(os.path.join(PROC, "X_val_technical.npy"),   feat_array[val_mask])
np.save(os.path.join(PROC, "X_test_technical.npy"),  feat_array[test_mask])

meta = {
    "tickers"      : ticker_index,
    "feature_names": FEATURE_NAMES,
    "n_dates"      : n_dates,
    "n_tickers"    : n_tickers,
    "n_features"   : N_FEATURES,
    "trading_days" : [str(d)[:10] for d in trading_days]
}
joblib.dump(meta, os.path.join(PROC, "stock_features_meta.pkl"))

print(f"\nX_train_technical : {feat_array[train_mask].shape}")
print(f"X_val_technical   : {feat_array[val_mask].shape}")
print(f"X_test_technical  : {feat_array[test_mask].shape}")
print("\nP5 ✅ — stock_features.npy + splits + scaler saved")