# pipeline/features/p8_merge_sentiment.py

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import RobustScaler

PROC = Path("data/processed")

# ── Step 1: Load daily_sentiment.csv ─────────────────────────────────────────
print("── Step 1: Loading daily_sentiment.csv ──")
sentiment = pd.read_csv(PROC / "daily_sentiment.csv")
sentiment["date"] = pd.to_datetime(sentiment["date"])

SENTIMENT_COLS = [
    "daily_score", "daily_pos_count", "daily_neg_count",
    "sentiment_5dma", "sentiment_vol", "sentiment_momentum", "sentiment_available"
]

print(f"Sentiment rows: {len(sentiment)}")
print(f"Sentiment cols: {sentiment.columns.tolist()}")
print(f"Nulls: {sentiment.isnull().sum().sum()}")

# ── Step 2: Load and update each split ───────────────────────────────────────
print("\n── Step 2: Merging sentiment into parquets ──")

splits = {}
for split in ["train", "val", "test"]:
    df = pd.read_parquet(PROC / f"{split}.parquet")
    
    # Confirm date is in index or column
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    
    before_shape = df.shape
    
    # Drop existing zero sentiment columns
    cols_to_drop = [c for c in SENTIMENT_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  {split}: dropped {len(cols_to_drop)} zero-sentiment cols | shape {before_shape} → {df.shape}")
    
    # Left-join real sentiment on date
    df = df.merge(sentiment[["date"] + SENTIMENT_COLS], on="date", how="left")
    
    # Verify no nulls introduced (all dates should match trading calendar)
    null_count = df[SENTIMENT_COLS].isnull().sum().sum()
    if null_count > 0:
        print(f"  ⚠️  {split}: {null_count} nulls after merge — filling with 0")
        df[SENTIMENT_COLS] = df[SENTIMENT_COLS].fillna(0)
        df["sentiment_available"] = df["sentiment_available"].fillna(0).astype(int)
    
    print(f"  {split}: final shape {df.shape} | nulls: {df.isnull().sum().sum()}")
    print(f"  {split}: sentiment_available=1: {(df['sentiment_available']==1).sum()} | =0: {(df['sentiment_available']==0).sum()}")
    
    splits[split] = df

# ── Step 3: Refit RobustScaler on train only (W3) ────────────────────────────
print("\n── Step 3: Refitting RobustScaler on train only (W3) ──")

train_df = splits["train"]

# Identify feature columns to scale — everything except date, target, and flag
NON_SCALE_COLS = ["date", "y_next_day_return", "sentiment_available",
                  "daily_pos_count", "daily_neg_count"]
feature_cols = [c for c in train_df.columns if c not in NON_SCALE_COLS]

print(f"Feature cols to scale: {len(feature_cols)}")
print(f"Feature cols: {feature_cols}")

# Fit scaler on train features only
scaler = RobustScaler()
train_features = train_df[feature_cols].values
scaler.fit(train_features)

print(f"Scaler fitted on train: {len(train_df)} rows")
print(f"Train mean (post-scale check): {scaler.center_[:5]}  ← should be ~0")

# ── Step 4: Apply scaler to all splits ───────────────────────────────────────
print("\n── Step 4: Applying scaler to all splits ──")

for split_name, df in splits.items():
    df_out = df.copy()
    df_out[feature_cols] = scaler.transform(df_out[feature_cols].values)
    
    # Restore non-scaled cols as-is (they're already in df_out)
    
    # Set date as index to match original parquet format
    df_out = df_out.set_index("date")
    
    splits[split_name] = df_out
    
    print(f"  {split_name}: scaled {len(feature_cols)} features | "
          f"mean={df_out[feature_cols].mean().mean():.6f} | "
          f"std={df_out[feature_cols].std().mean():.6f}")

# ── Step 5: Save parquets and scaler ─────────────────────────────────────────
print("\n── Step 5: Saving outputs ──")

for split_name, df in splits.items():
    out_path = PROC / f"{split_name}.parquet"
    df.to_parquet(out_path)
    print(f"  ✅ {split_name}.parquet saved | shape: {df.shape}")

with open(PROC / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print(f"  ✅ scaler.pkl saved (refit on new train with real sentiment)")

# ── Step 6: Validation ────────────────────────────────────────────────────────
print("\n── Validation ───────────────────────────────────────────────────────")

train_check = splits["train"]
val_check   = splits["val"]
test_check  = splits["test"]

print(f"Train shape:  {train_check.shape} | nulls: {train_check.isnull().sum().sum()}")
print(f"Val shape:    {val_check.shape}   | nulls: {val_check.isnull().sum().sum()}")
print(f"Test shape:   {test_check.shape}  | nulls: {test_check.isnull().sum().sum()}")

print(f"\nSentiment columns in train (first 5 rows):")
print(train_check[SENTIMENT_COLS].head().to_string())

print(f"\nSentiment in val — should have flag=1 (2021-2022):")
print(val_check[["daily_score","sentiment_available"]].head(5).to_string())

print(f"\nSentiment in test — should have flag=1 (2023-2025):")
print(test_check[["daily_score","sentiment_available"]].head(5).to_string())

# Confirm scaler.pkl is the NEW one (not the old zero-sentiment one)
with open(PROC / "scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)
print(f"\nScaler center (median) for first 3 features: {loaded_scaler.center_[:3]}")
print(f"Scaler scale (IQR) for first 3 features:    {loaded_scaler.scale_[:3]}")
print(f"Scaler n_features: {loaded_scaler.n_features_in_}")

print("\n✅ P8 Complete — train/val/test.parquet and scaler.pkl updated with real sentiment")
print("   Next: P6 rerun (same script, same config, new data)")