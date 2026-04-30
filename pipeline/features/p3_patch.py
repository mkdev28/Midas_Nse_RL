# pipeline/features/p3_patch_c1_final.py
"""
Patches features_v1.parquet — adds 3 C1 features only.
Run this, then P4, then P8. Nothing else changes.
"""

import os, sys
import pandas as pd
import numpy as np
sys.path.append(r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse")
from config import PROC

# ── Load features_v1 (all technical features already present) ────────────────
df = pd.read_parquet(os.path.join(PROC, "features_v1.parquet"))
print(f"Loaded features_v1: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")

# Drop if somehow already exist from failed patch
for c in ["dii_absorption_ratio", "vix_regime", "institutional_net"]:
    if c in df.columns:
        df = df.drop(columns=[c])
        print(f"  Dropped existing {c}")

# ── C1a: DII Absorption Ratio ─────────────────────────────────────────────────
fii_selling = df["fii_net"] < 0
df["dii_absorption_ratio"] = np.where(
    fii_selling,
    np.clip(df["dii_net"] / (df["fii_net"].abs() + 1e-9), -3.0, 3.0),
    0.0
)

# ── C1b: VIX Regime ───────────────────────────────────────────────────────────
df["vix_regime"] = np.select(
    [df["vix_close"] < 15,
     (df["vix_close"] >= 15) & (df["vix_close"] < 25),
     df["vix_close"] >= 25],
    [0, 1, 2], default=1
).astype(float)

# ── C1c: Institutional Net ────────────────────────────────────────────────────
df["institutional_net"] = df["fii_net"] + df["dii_net"]

# ── Verify ────────────────────────────────────────────────────────────────────
new_nulls = df[["dii_absorption_ratio","vix_regime","institutional_net"]].isnull().sum().sum()
assert new_nulls == 0, f"Nulls in C1 features: {new_nulls}"
assert df.isnull().sum().sum() == 0, "Nulls elsewhere after patch"

print(f"\nC1 features added:")
print(f"  dii_absorption_ratio: mean={df['dii_absorption_ratio'].mean():.4f} | FII selling days: {fii_selling.sum()}")
print(f"  vix_regime: calm={( df['vix_regime']==0).sum()} | elevated={(df['vix_regime']==1).sum()} | crisis={(df['vix_regime']==2).sum()}")
print(f"  institutional_net: mean={df['institutional_net'].mean():.2f}")
print(f"\nFinal shape: {df.shape}")
print(f"All columns: {df.columns.tolist()}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(os.path.join(PROC, "features_v1.parquet"), index=False)
print(f"\n✅ features_v1.parquet patched and saved")
print(f"   Now run: python pipeline/features/p4_split_normalize.py")
print(f"   Then run: python pipeline/features/p8_merge_sentiment.py")