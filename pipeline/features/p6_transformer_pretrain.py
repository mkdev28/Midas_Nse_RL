# pipeline/features/p6_transformer_pretrain.py
"""
P6 Rerun — Transformer Pretrain with real sentiment (43 features)
Change from original: Linear(36→256) to Linear(43→256)
Same config, same script, new data.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math

# ── Config (locked — do not change) ──────────────────────────────────────────
PROC        = Path("data/processed")
CKPT        = Path("checkpoints")
WINDOW      = 60
D_MODEL     = 256
N_HEADS     = 4
N_LAYERS    = 3
D_FF        = 512
DROPOUT     = 0.1
BATCH_SIZE  = 256
LR          = 1e-3
PATIENCE    = 15
MAX_EPOCHS  = 200
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

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
TARGET_COL  = "y_next_day_return"
N_FEATURES  = len(FEATURE_COLS)   # 43

print(f"Device: {DEVICE}")
print(f"Features: {N_FEATURES}")
assert N_FEATURES == 46, f"Expected 46 features, got {N_FEATURES}"

CKPT.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.window]           # (60, 43)
        y = self.y[idx + self.window]                 # scalar target
        return torch.tensor(x), torch.tensor(y)


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Transformer Model ─────────────────────────────────────────────────────────
class TransformerPretrainer(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)   # Linear(46, 256)
        self.pos_enc    = PositionalEncoding(d_model, dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.head       = nn.Linear(d_model, 1)             # predict next-day return

    def encode(self, x):
        """x: (batch, 60, 43) → z: (batch, 256)"""
        x = self.input_proj(x)   # → (batch, 60, 256)
        x = self.pos_enc(x)      # → (batch, 60, 256)
        x = self.encoder(x)      # → (batch, 60, 256)
        x = self.norm(x)
        return x[:, -1, :]       # → (batch, 256) last timestep

    def forward(self, x):
        z = self.encode(x)
        return self.head(z).squeeze(-1)   # → (batch,)


# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading parquets...")
train_df = pd.read_parquet(PROC / "train.parquet").reset_index()
val_df   = pd.read_parquet(PROC / "val.parquet").reset_index()

# Verify all feature cols exist
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

# ── Model, optimizer, scheduler ───────────────────────────────────────────────
model = TransformerPretrainer(N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
criterion = nn.MSELoss()

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nStarting training...")
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

    print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}"
          + (" ← best" if val_loss < best_val_loss else ""))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save full checkpoint
        torch.save({
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "val_loss":     val_loss,
            "feature_cols": FEATURE_COLS,
            "config": {
                "n_features": N_FEATURES,
                "d_model":    D_MODEL,
                "n_heads":    N_HEADS,
                "n_layers":   N_LAYERS,
                "d_ff":       D_FF,
                "dropout":    DROPOUT,
                "window":     WINDOW,
            }
        }, CKPT / "transformer_best.pt")
        # Save encoder only (what RL agents load)
        torch.save({
            "encoder_state": {k: v for k, v in model.state_dict().items()
                              if not k.startswith("head.")},
            "feature_cols":  FEATURE_COLS,
            "config": {"n_features": N_FEATURES, "d_model": D_MODEL,
                       "n_heads": N_HEADS, "n_layers": N_LAYERS,
                       "d_ff": D_FF, "dropout": DROPOUT, "window": WINDOW}
        }, CKPT / "transformer_encoder.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# ── Save training history ─────────────────────────────────────────────────────
pd.DataFrame(history).to_csv(CKPT / "pretrain_history.csv", index=False)

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n✅ P6 Rerun Complete")
print(f"   Best val loss: {best_val_loss:.6f}")
print(f"   Epochs run:    {len(history)}")
print(f"   transformer_best.pt    → {CKPT / 'transformer_best.pt'}")
print(f"   transformer_encoder.pt → {CKPT / 'transformer_encoder.pt'}")

# Verify saved checkpoint
ckpt = torch.load(CKPT / "transformer_best.pt", map_location="cpu", weights_only=False)
print(f"\nCheckpoint verification:")
print(f"   feature_cols count: {len(ckpt['feature_cols'])}")
print(f"   n_features in config: {ckpt['config']['n_features']}")
print(f"   val_loss: {ckpt['val_loss']:.6f}")
assert len(ckpt["feature_cols"]) == 46, "feature_cols mismatch in saved checkpoint"
print("   ✅ Checkpoint verified — 46 features confirmed")