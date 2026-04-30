# pipeline/features/p9_alignment_check.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

PROC = Path("data/processed")

train = pd.read_parquet(PROC / "train.parquet").reset_index()
train["date"] = pd.to_datetime(train["date"])

with open(PROC / "stock_features_meta.pkl", "rb") as f:
    meta = pickle.load(f)

print(f"train.parquet start: {train['date'].iloc[0].date()}")
print(f"train.parquet end:   {train['date'].iloc[-1].date()}")
print(f"train.parquet rows:  {len(train)}")
print(f"\nmeta keys: {list(meta.keys())}")

X_train = np.load(PROC / "X_train_technical.npy")
print(f"\nX_train_technical shape: {X_train.shape}")

if "train_dates" in meta:
    dates = pd.to_datetime(meta["train_dates"])
    print(f"X_train start: {dates[0].date()}")
    print(f"X_train end:   {dates[-1].date()}")
elif "dates" in meta:
    dates = pd.to_datetime(meta["dates"])
    print(f"All dates: {dates[0].date()} → {dates[-1].date()}")
    print(f"Total date rows: {len(dates)}")
else:
    print(f"meta contents: {meta}")