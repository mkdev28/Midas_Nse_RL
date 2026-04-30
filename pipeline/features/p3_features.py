# pipeline/features/p3_features.py
import sys, os
import pandas as pd
import numpy as np
sys.path.append(r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse")
from config import PROC

df = pd.read_parquet(os.path.join(PROC, "master_aligned.parquet"))
df = df.sort_values('date').reset_index(drop=True)

# ── Returns ──────────────────────────────────────────────────────────
df['ret_1d']  = df['nifty_close'].pct_change(1)
df['ret_5d']  = df['nifty_close'].pct_change(5)
df['ret_20d'] = df['nifty_close'].pct_change(20)

# ── RSI (14) ─────────────────────────────────────────────────────────
delta = df['nifty_close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss.where(loss != 0, other=1e-10)   # ← was: loss.replace(0, np.nan)
df['rsi_14'] = 100 - (100 / (1 + rs))
# ── MACD (12, 26, 9) ─────────────────────────────────────────────────
ema12             = df['nifty_close'].ewm(span=12, adjust=False).mean()
ema26             = df['nifty_close'].ewm(span=26, adjust=False).mean()
df['macd']        = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_hist']   = df['macd'] - df['macd_signal']

# ── Bollinger Bands (20, 2σ) ─────────────────────────────────────────
bb_mid         = df['nifty_close'].rolling(20).mean()
bb_std         = df['nifty_close'].rolling(20).std()
df['bb_upper'] = bb_mid + 2 * bb_std
df['bb_lower'] = bb_mid - 2 * bb_std
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
df['bb_pct']   = (df['nifty_close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# ── ATR (14) ─────────────────────────────────────────────────────────
high, low, close_prev = df['nifty_high'], df['nifty_low'], df['nifty_close'].shift(1)
tr = pd.concat([high - low,
                (high - close_prev).abs(),
                (low  - close_prev).abs()], axis=1).max(axis=1)
df['atr_14']  = tr.rolling(14).mean()
df['atr_pct'] = df['atr_14'] / df['nifty_close']

# ── Macro Spreads ─────────────────────────────────────────────────────
df['yield_spread']     = df['gsec_10y_yield'] - df['repo_rate']
df['gold_crude_ratio'] = df['gold_close'] / df['crude_close']

# ── FII/DII Momentum ─────────────────────────────────────────────────
df['fii_net_5d']     = df['fii_net'].rolling(5).sum()
df['fii_net_20d']    = df['fii_net'].rolling(20).sum()
df['dii_net_5d']     = df['dii_net'].rolling(5).sum()
df['dii_net_20d']    = df['dii_net'].rolling(20).sum()
df['fii_dii_net_5d'] = df['fii_net_5d'] + df['dii_net_5d']

# ── TARGET ───────────────────────────────────────────────────────────
df['y_next_day_return'] = df['nifty_close'].shift(-1) / df['nifty_close'] - 1
df = df.iloc[:-1].reset_index(drop=True)   # drop last row (NaN target)

# ── CLEANUP (must happen before asserts) ─────────────────────────────
df.drop(columns=['nifty_volume', 'volume_ma20', 'volume_ratio'],
        inplace=True, errors='ignore')      # errors='ignore' handles missing cols
df = df.iloc[26:].reset_index(drop=True)   # drop MACD warmup rows
# DEBUG — add this before the assert
remaining_nulls = df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
print("Remaining nulls after warmup drop:")
print(remaining_nulls)
print("\nSample rows with nulls:")
print(df[df.isnull().any(axis=1)][['date'] + remaining_nulls.index.tolist()].head(10).to_string())
# ── FINAL ASSERTS ─────────────────────────────────────────────────────
assert df['y_next_day_return'].isnull().sum() == 0, "NaN found in y!"
assert df.isnull().sum().sum() == 0, "Still has nulls!"

print(f"Final shape  : {df.shape}")        # expect (4387, 45)
print(f"Date range   : {df['date'].min()} → {df['date'].max()}")
print(f"Columns      : {df.columns.tolist()}")

df.to_parquet(os.path.join(PROC, "features_v1.parquet"), index=False)
print("\nP3 ✅ — features_v1.parquet saved")