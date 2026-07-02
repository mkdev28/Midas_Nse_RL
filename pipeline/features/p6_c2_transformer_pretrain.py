# pipeline/features/p6_c2_transformer_pretrain.py
"""
P6 C2 — Pretrain CrossModalTransformer on next-day return prediction.

Identical task to p6_transformer_pretrain.py (MSE on y_next_day_return),
but uses CrossModalTransformer instead of unified Transformer.

Key differences from p6_transformer_pretrain.py:
  - Uses CrossModalTransformerPretrainer (3 modality encoders + cross-modal Transformer)
  - Checkpoints saved to checkpoints/c2/  (NEVER overwrites unified checkpoints)
  - TensorBoard logs to logs/p6_c2/
  - Saves architecture tag "c2_cross_modal" in checkpoint

Do NOT modify this file to touch checkpoints/transformer_*.pt —
those are the unified Transformer results and must be preserved for ablation.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from models.cross_modal_transformer import CrossModalTransformerPretrainer

# ── Config (locked) ───────────────────────────────────────────────────────────
PROC       = Path("data/processed")
CKPT_C2    = Path("checkpoints/c2")          # C2 checkpoints — isolated
LOGS_C2    = Path("logs/p6_c2")
WINDOW     = 60
D_OUT      = 256                              # must match unified (agent obs unchanged)
BATCH_SIZE = 256
LR         = 1e-3
PATIENCE   = 15
MAX_EPOCHS = 200
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Feature columns — must match unified Transformer exactly
FEATURE_COLS = [
    "nifty_close", "nifty_high", "nifty_low", "nifty_open",
    "vix_close", "gold_close", "crude_close", "inrusd_close",
    "repo_rate", "gsec_10y_yield", "yield_spread", "gold_crude_ratio",
    "fii_net", "dii_net", "fii_buy_value", "fii_sell_value",
    "dii_buy_value", "dii_sell_value",
    "fii_net_5d", "fii_net_20d", "dii_net_5d", "dii_net_20d", "fii_dii_net_5d",
    "ret_1d", "ret_5d", "ret_20d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "atr_14", "atr_pct",
    "dii_absorption_ratio", "vix_regime", "institutional_net",
    "daily_score", "daily_pos_count", "daily_neg_count",
    "sentiment_5dma", "sentiment_vol", "sentiment_momentum", "sentiment_available"
]
TARGET_COL = "y_next_day_return"
N_FEATURES = len(FEATURE_COLS)

assert N_FEATURES == 46, f"Expected 46 features, got {N_FEATURES}"

CKPT_C2.mkdir(parents=True, exist_ok=True)
LOGS_C2.mkdir(parents=True, exist_ok=True)

print(f"Device  : {DEVICE}")
print(f"Features: {N_FEATURES}")
print(f"C2 checkpoints → {CKPT_C2}")
print(f"Unified checkpoints → checkpoints/ (UNTOUCHED)")


# ── Dataset (identical to p6_transformer_pretrain.py) ─────────────────────────
class WindowDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.window]          # (60, 46)
        y = self.y[idx + self.window]                # scalar
        return torch.tensor(x), torch.tensor(y)


# ── Load data ──────────────────────────────────────────────────────────────────
print("\nLoading parquets...")
train_df = pd.read_parquet(PROC / "train.parquet").reset_index()
val_df   = pd.read_parquet(PROC / "val.parquet").reset_index()

missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in train_df.columns]
assert len(missing) == 0, f"Missing columns: {missing}"

print(f"Train: {len(train_df)} rows | Val: {len(val_df)} rows")
print(f"Train nulls: {train_df[FEATURE_COLS].isnull().sum().sum()}")
print(f"Val nulls:   {val_df[FEATURE_COLS].isnull().sum().sum()}")

train_ds = WindowDataset(train_df, FEATURE_COLS, TARGET_COL, WINDOW)
val_ds   = WindowDataset(val_df,   FEATURE_COLS, TARGET_COL, WINDOW)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")


# ── Model ─────────────────────────────────────────────────────────────────────
model = CrossModalTransformerPretrainer(window=WINDOW, d_out=D_OUT)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters: {total_params:,}")

# Sanity check: verify output shape
_x = torch.randn(2, WINDOW, N_FEATURES).to(DEVICE)
_z = model.encoder(_x)
assert _z.shape == (2, D_OUT), f"Z shape mismatch: {_z.shape}"
print(f"Z shape verified: {_z.shape} ✅")

# Verify attention weights extractable
_attn = model.encoder.get_attention_weights(_x)
assert len(_attn) == 3, f"Expected 3 attention layers, got {len(_attn)}"
expected_seq = WINDOW * 3   # 180 tokens (60 per modality × 3)
assert _attn[0].shape[-1] == expected_seq, f"Attention shape wrong: {_attn[0].shape}"
print(f"Attention shape verified: {_attn[0].shape} ✅  (seq_len={expected_seq})")
del _x, _z, _attn


# ── Train ─────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
criterion = nn.MSELoss()

print("\nStarting C2 Transformer pretraining...")
history = []
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            val_losses.append(criterion(pred, y_batch).item())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    scheduler.step()
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    is_best = val_loss < best_val_loss
    print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}"
          + (" ← best" if is_best else ""))

    if is_best:
        best_val_loss = val_loss
        patience_counter = 0

        # Save full C2 checkpoint (includes encoder + head weights)
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "val_loss":     val_loss,
            "feature_cols": FEATURE_COLS,
            "architecture": "c2_cross_modal",
            "config": {
                "n_features":  N_FEATURES,
                "d_out":       D_OUT,
                "window":      WINDOW,
                "n_layers":    3,
                "nhead":       4,
                "d_enc":       128,
                "d_ff":        512,
                "dropout":     0.1,
            }
        }, CKPT_C2 / "transformer_c2_best.pt")

        # Save encoder only (what RL agents load — drop-in for transformer_encoder.pt)
        encoder_state = {k: v
                         for k, v in model.state_dict().items()
                         if not k.startswith("head.")}
        torch.save({
            "encoder_state": encoder_state,
            "feature_cols":  FEATURE_COLS,
            "architecture":  "c2_cross_modal",
            "config": {
                "n_features": N_FEATURES, "d_out": D_OUT,
                "window": WINDOW, "n_layers": 3,
                "nhead": 4, "d_enc": 128,
            }
        }, CKPT_C2 / "transformer_c2_encoder.pt")

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# ── Save training history ──────────────────────────────────────────────────────
pd.DataFrame(history).to_csv(CKPT_C2 / "pretrain_c2_history.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
# Load unified val_loss for comparison
unified_ckpt = torch.load("checkpoints/transformer_best.pt",
                           map_location="cpu", weights_only=False)
unified_val_loss = unified_ckpt.get("val_loss", float("nan"))

print(f"\n{'='*60}")
print(f"C2 Pretraining Complete")
print(f"  Best val loss (C2):     {best_val_loss:.6f}")
print(f"  Best val loss (unified): {unified_val_loss:.6f}")
ratio = best_val_loss / (unified_val_loss + 1e-12)
print(f"  C2/unified ratio:        {ratio:.2f}x")
print(f"  Epochs run:              {len(history)}")
print(f"")
print(f"  C2 full checkpoint  → {CKPT_C2 / 'transformer_c2_best.pt'}")
print(f"  C2 encoder only     → {CKPT_C2 / 'transformer_c2_encoder.pt'}")
print(f"  C2 history          → {CKPT_C2 / 'pretrain_c2_history.csv'}")
print(f"")
print(f"  Unified checkpoints → checkpoints/transformer_*.pt  (UNTOUCHED ✅)")
print(f"{'='*60}")

# Verify saved C2 checkpoint
ckpt = torch.load(CKPT_C2 / "transformer_c2_best.pt", map_location="cpu", weights_only=False)
assert ckpt["architecture"] == "c2_cross_modal"
assert len(ckpt["feature_cols"]) == 46
print(f"\nCheckpoint verified — architecture={ckpt['architecture']} | features={len(ckpt['feature_cols'])} ✅")
