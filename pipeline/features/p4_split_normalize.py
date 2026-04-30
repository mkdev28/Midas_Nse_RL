# pipeline/features/p4_split_normalize.py
import sys, os
import pandas as pd
import numpy as np
import joblib
sys.path.append(r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse")
from config import PROC

df = pd.read_parquet(os.path.join(PROC, "features_v1.parquet"))
df['date'] = pd.to_datetime(df['date'])

# ── Splits (time-ordered, no shuffling) ──────────────────────────────
train = df[df['date'] <  '2021-01-01'].reset_index(drop=True)
val   = df[(df['date'] >= '2021-01-01') & (df['date'] < '2023-01-01')].reset_index(drop=True)
test  = df[df['date'] >= '2023-01-01'].reset_index(drop=True)

print(f"Train : {len(train)} rows  {train['date'].min().date()} → {train['date'].max().date()}")
print(f"Val   : {len(val)}  rows  {val['date'].min().date()} → {val['date'].max().date()}")
print(f"Test  : {len(test)}  rows  {test['date'].min().date()} → {test['date'].max().date()}")

# ── Columns to normalize ──────────────────────────────────────────────
# Exclude: date, calendar cols, flags, and target
DO_NOT_SCALE = ['date', 'year', 'month', 'quarter', 'dayofweek',
                'weekofyear', 't_index', 'fii_dii_available', 'y_next_day_return']
feature_cols = [c for c in df.columns if c not in DO_NOT_SCALE]

# ── Fit scaler on TRAIN only ──────────────────────────────────────────
means = train[feature_cols].mean()
stds  = train[feature_cols].std().replace(0, 1)   # avoid div-by-zero for constant cols

def normalize(split_df):
    out = split_df.copy()
    out[feature_cols] = (split_df[feature_cols] - means) / stds
    return out

train_norm = normalize(train)
val_norm   = normalize(val)
test_norm  = normalize(test)

# ── Save splits ───────────────────────────────────────────────────────
train_norm.to_parquet(os.path.join(PROC, "train.parquet"), index=False)
val_norm.to_parquet(os.path.join(PROC,   "val.parquet"),   index=False)
test_norm.to_parquet(os.path.join(PROC,  "test.parquet"),  index=False)

# Save scaler stats for inference time
scaler = {'means': means, 'stds': stds, 'feature_cols': feature_cols}
joblib.dump(scaler, os.path.join(PROC, "scaler.pkl"))

# ── Verify ────────────────────────────────────────────────────────────
print(f"\nTrain feature mean (should be ~0): {train_norm[feature_cols].mean().mean():.6f}")
print(f"Train feature std  (should be ~1): {train_norm[feature_cols].std().mean():.6f}")
print(f"Val   nulls: {val_norm.isnull().sum().sum()}")
print(f"Test  nulls: {test_norm.isnull().sum().sum()}")

print("\nP4 ✅ — train/val/test.parquet + scaler.pkl saved")