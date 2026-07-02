import pandas as pd
import pickle
import numpy as np

# --- Blocker 1: W7 — Resolve 26-row offset ---
train = pd.read_parquet("data/processed/train.parquet")
print("train.parquet start date :", train.index[0])
print("train.parquet end date   :", train.index[-1])
print("train.parquet shape      :", train.shape)

# --- Blocker 2: Confirm Agent 3 obs dim (856 or 844?) ---
with open("data/processed/stock_features_meta.pkl", "rb") as f:
    meta = pickle.load(f)
print("\nstock_features_meta:", meta)

X_train = np.load("data/processed/X_train_technical.npy")
print("X_train_technical shape  :", X_train.shape)
print("X_train_technical start  :", X_train.shape[0], "rows")

# --- Bonus: Verify encoder feature cols match parquet ---
import torch
ckpt = torch.load("checkpoints/transformer_encoder.pt", weights_only=False)
print("\nEncoder feature_cols count:", len(ckpt['feature_cols']))
print("Feature cols:", ckpt['feature_cols'])