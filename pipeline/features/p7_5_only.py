# pipeline/features/p7_5_only.py
# Run this standalone — fine-tuning and inference already done

import pandas as pd
import numpy as np
from pathlib import Path

PROC  = Path("data/processed")
RAW   = Path("data/raw")
SENTIMENT_START = pd.Timestamp("2014-01-01")

# ── Load trading calendar ─────────────────────────────────────────────────────
cal = pd.read_parquet(PROC / "trading_calendar.parquet")
cal["date"] = pd.to_datetime(cal["date"])
full = cal[["date"]].copy().sort_values("date").reset_index(drop=True)

# ── Load VIX — handles MultiIndex columns from yfinance ──────────────────────
vix_raw = pd.read_parquet(RAW / "india_vix.parquet")
print(f"VIX raw shape: {vix_raw.shape}")
print(f"VIX columns: {vix_raw.columns.tolist()[:6]}")
print(f"VIX index type: {type(vix_raw.index)}")

# Flatten MultiIndex columns if present
if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = ['_'.join([str(c) for c in col if c]).strip('_')
                       for col in vix_raw.columns]
    print(f"Flattened columns: {vix_raw.columns.tolist()}")

# Reset index — date is likely the index in yfinance parquets
vix_raw = vix_raw.reset_index()
print(f"After reset_index columns: {vix_raw.columns.tolist()}")

# Find date column
date_col = next((c for c in vix_raw.columns
                 if str(c).lower() in ("date", "datetime", "index")), vix_raw.columns[0])

# Find Close column (not Adj Close)
close_col = next((c for c in vix_raw.columns
                  if "close" in str(c).lower() and "adj" not in str(c).lower()), None)
if close_col is None:
    # Fallback: first numeric column
    close_col = vix_raw.select_dtypes(include=np.number).columns[0]

print(f"Using date_col='{date_col}', close_col='{close_col}'")

vix = vix_raw[[date_col, close_col]].copy()
vix.columns = ["date", "vix_close"]
vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
vix = vix.sort_values("date").drop_duplicates("date").reset_index(drop=True)
vix["vix_20dma"] = vix["vix_close"].rolling(20, min_periods=5).mean()
vix["vix_pseudo"] = np.clip(
    -(vix["vix_close"] - vix["vix_20dma"]) / vix["vix_20dma"], -1.0, 1.0
)
print(f"VIX rows: {len(vix)} | {vix['date'].min()} → {vix['date'].max()}")
print(f"VIX pseudo range: {vix['vix_pseudo'].min():.4f} → {vix['vix_pseudo'].max():.4f}")

full = full.merge(vix[["date", "vix_pseudo"]], on="date", how="left")
full["vix_pseudo"] = full["vix_pseudo"].fillna(0)

# ── Load FinBERT RSS results (already computed in P7.4) ───────────────────────
df_headline_sentiment = pd.read_csv(PROC / "headline_sentiment.csv")
print(f"\nLoaded headline_sentiment.csv: {len(df_headline_sentiment)} rows")

df_dated = df_headline_sentiment.dropna(subset=["date"]).copy()
df_dated["date"] = pd.to_datetime(df_dated["date"]).dt.normalize()
df_dated["sentiment_score"] = df_dated["pos"] - df_dated["neg"]

def get_dominant(row):
    if row["pos"] >= row["neg"] and row["pos"] >= row["neu"]:
        return "pos"
    elif row["neg"] >= row["neu"]:
        return "neg"
    return "neu"

df_dated["dominant"] = df_dated.apply(get_dominant, axis=1)

finbert_score = (df_dated.groupby("date")["sentiment_score"]
                 .mean().reset_index()
                 .rename(columns={"sentiment_score": "finbert_score"}))

counts = (df_dated.groupby(["date", "dominant"])
          .size().unstack(fill_value=0).reset_index())
counts.columns.name = None
for col in ["pos", "neg", "neu"]:
    if col not in counts.columns:
        counts[col] = 0
counts = counts.rename(columns={"pos": "daily_pos_count", "neg": "daily_neg_count"})

finbert_daily = finbert_score.merge(
    counts[["date", "daily_pos_count", "daily_neg_count"]], on="date", how="left"
)
full = full.merge(finbert_daily, on="date", how="left")

# ── Combine tiers ─────────────────────────────────────────────────────────────
full["daily_score"]     = np.where(full["finbert_score"].notna(),
                                   full["finbert_score"], full["vix_pseudo"])
full["daily_pos_count"] = full["daily_pos_count"].fillna(0).astype(int)
full["daily_neg_count"] = full["daily_neg_count"].fillna(0).astype(int)
full["sentiment_available"] = (full["date"] >= SENTIMENT_START).astype(int)

# Hard enforce pre-2014
neutral_cols = ["daily_score", "daily_pos_count", "daily_neg_count"]
mask_pre = full["date"] < SENTIMENT_START
full.loc[mask_pre, "daily_score"]      = full.loc[mask_pre, "vix_pseudo"]
full.loc[mask_pre, "daily_pos_count"]  = 0
full.loc[mask_pre, "daily_neg_count"]  = 0
full.loc[mask_pre, "sentiment_available"] = 0

# ── Rolling features ──────────────────────────────────────────────────────────
full = full.sort_values("date").reset_index(drop=True)
full["sentiment_5dma"]     = full["daily_score"].rolling(5, min_periods=1).mean()
full["sentiment_vol"]      = full["daily_score"].rolling(5, min_periods=2).std().fillna(0)
full["sentiment_momentum"] = full["daily_score"] - full["sentiment_5dma"]

# ── Final output ──────────────────────────────────────────────────────────────
out_cols = ["date", "daily_score", "daily_pos_count", "daily_neg_count",
            "sentiment_5dma", "sentiment_vol", "sentiment_momentum", "sentiment_available"]
full = full[out_cols]

# ── Validation ────────────────────────────────────────────────────────────────
print("\n── Validation ───────────────────────────────────────────────────────")
print(f"Total rows:             {len(full)}")
print(f"sentiment_available=1:  {(full['sentiment_available']==1).sum()}")
print(f"sentiment_available=0:  {(full['sentiment_available']==0).sum()}")
print(f"Nulls remaining:        {full.isnull().sum().sum()}")
print(f"daily_score range:      {full['daily_score'].min():.4f} → {full['daily_score'].max():.4f}")

print(f"\nSanity — 2008 GFC (expect negative VIX proxy):")
print(full[(full["date"]>="2008-09-01") & (full["date"]<="2008-11-30")]
      [["date","daily_score","sentiment_available"]].head(6).to_string())

print(f"\nSanity — 2020 COVID crash (expect negative):")
print(full[(full["date"]>="2020-02-15") & (full["date"]<="2020-04-15")]
      [["date","daily_score","sentiment_available"]].head(6).to_string())

print(f"\nSanity — recent RSS days (FinBERT):")
print(full[full["daily_pos_count"] > 0].tail(10)
      [["date","daily_score","daily_pos_count","daily_neg_count","sentiment_available"]].to_string())

full.to_csv(PROC / "daily_sentiment.csv", index=False)
print(f"\n✅ daily_sentiment.csv saved → {PROC / 'daily_sentiment.csv'}")
print(f"   Shape: {full.shape}")
print(f"   Columns: {full.columns.tolist()}")