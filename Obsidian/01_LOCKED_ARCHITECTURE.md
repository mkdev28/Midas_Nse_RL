Here is the fully corrected `01_LOCKED_ARCHITECTURE.md`:

***

```markdown
# MIDAS-NSE — Locked Architecture

> **Status:** 🔒 LOCKED — Do not modify unless explicitly decided by the project owner.
> **Last reviewed:** 2026-04-30

---

## System Overview

```
RAW DATA (pre-computed, no API calls during training)
  ├── Price/Technical : yfinance OHLCV for 50 .NS tickers
  ├── Macro signals   : yfinance (VIX, INR/USD, Gold, Crude) + NSE scrape (FII/DII)
  └── News sentiment  : FinBERT fine-tuned on kdave/Indian_Financial_News
                        (F1=0.917) + VIX pseudo-sentiment proxy pre-2014
         ↓
SHARED TRANSFORMER ENCODER (pretrained, frozen during RL)
  Input  : (batch, 60 days, 46 features)
  Output : Z — (batch, 256) — compressed market state embedding
         ↓
THREE RL AGENTS (all receive Z)
         ↓
PURE MATH COORDINATOR (deterministic)
         ↓
XAI LAYER (SHAP + attention attribution)
```

---

## Five Research Contributions (Do Not Alter)

### C1 — India-specific Hierarchical DRL
Agent 1 (SAC, Calmar reward) decides macro asset class weights. Agent 3 (SAC, Sharpe minus correlation penalty) picks stocks. The **DII Absorption Ratio** — `dii_net / abs(fii_net)` on days when `fii_net < 0` — is the key India-unique signal detecting the 2022-2024 regime shift.

### C2 — Modality-specialized Encoders + Cross-modal Transformer Fusion
- Macro → MLP encoder
- Sentiment → Linear projection encoder
- Technical/OHLCV → 1D CNN encoder
- All produce same-dimension tokens → fuse via shared cross-modal Transformer with modality type embeddings

> Note: Current implementation uses a unified Transformer over all 46 features.
> Modality-specialized encoders (C2) are the target architecture for the RL
> training phase — not yet implemented.

### C3 — Per-decision XAI with Cross-modal Attribution
SHAP on Agent 1 and Agent 3 + Transformer attention weights → explains *why* a specific allocation was made on a specific day.

### C4 — NSE-specific Latency Benchmarking
Profile with `torch.profiler`. Prove total pipeline < 500ms. Compare against Gao et al. (~15.95s). First multi-agent multimodal benchmark on NSE.

### C5 — Selective Replay Buffer for Indian Market Regime Shifts
Tail-risk episodes (VIX > 25, FII net z-score > 2) permanently retained. Normal experiences FIFO. Forces agents to revisit 2008 GFC, 2013 Taper Tantrum, 2020 COVID crash, 2022 FII exodus.
Normal buffer capacity: 100,000, FIFO
Tail-risk buffer: unlimited, never discarded
Sample ratio per batch: 80% normal / 20% tail-risk

---

## Transformer Encoder

**Config (locked):**
```
WINDOW       = 60 days
D_MODEL      = 256
N_HEADS      = 4
N_LAYERS     = 3
D_FF         = 512
DROPOUT      = 0.1
BATCH_SIZE   = 256
LR           = 1e-3
PATIENCE     = 15
Scheduler    = CosineAnnealingLR
Loss         = MSELoss
Optimizer    = AdamW (weight_decay=1e-4)
Grad clip    = 1.0
Input proj   = Linear(46, 256)
```

**encode() method:**
```python
def encode(self, x):       # x: (batch, 60, 46)
    x = self.input_proj(x) # → (batch, 60, 256)
    x = self.pos_enc(x)    # → (batch, 60, 256)
    x = self.encoder(x)    # → (batch, 60, 256)
    x = self.norm(x)
    return x[:, -1, :]     # → (batch, 256) — last timestep
```

**Current result:** Best val loss 0.000098 at epoch 20. Early stopped at epoch 35.
✅ Retrained 2026-04-30 with 46 features including real FinBERT sentiment + 3 C1 features.

---

## Transformer Input Features (46 columns, exact order)
✅ VERIFIED 2026-04-30 — ground truth is checkpoint['feature_cols'] in transformer_best.pt

**Price (4):**
nifty_close, nifty_high, nifty_low, nifty_open

**Macro (8):**
vix_close, gold_close, crude_close, inrusd_close,
repo_rate, gsec_10y_yield, yield_spread, gold_crude_ratio

**FII/DII Flows (11):**
fii_net, dii_net, fii_buy_value, fii_sell_value,
dii_buy_value, dii_sell_value,
fii_net_5d, fii_net_20d, dii_net_5d, dii_net_20d, fii_dii_net_5d

**Technical (13):**
ret_1d, ret_5d, ret_20d,
rsi_14, macd, macd_signal, macd_hist,
bb_upper, bb_lower, bb_width, bb_pct,
atr_14, atr_pct

**C1 India Signals (3):**
dii_absorption_ratio, vix_regime, institutional_net

**Sentiment (7):**
daily_score, daily_pos_count, daily_neg_count,
sentiment_5dma, sentiment_vol, sentiment_momentum, sentiment_available

**TARGET (not fed to Transformer):**
y_next_day_return

---

## Agent Specifications

### Agent 1 — SAC Macro Allocator

| Property | Value |
|----------|-------|
| Algorithm | SAC (Stable Baselines3) |
| State dim | 265 = Z(256) + 5 macro + 4 current weights |
| Macro signals | VIX, G-Sec yield, INR/USD, FII net, daily_score |
| Action | 4 class weights (stocks, bonds, commodities, cash) → softmax |
| Reward | Calmar ratio (annualized return / max drawdown, 60-day rolling) |
| Timesteps | 500,000 |

### Agent 2 — PPO Sentiment Modifier

| Property | Value |
|----------|-------|
| Algorithm | PPO (Stable Baselines3) |
| State dim | 7 = 3 sentiment signals + 4 frozen A1 weights |
| Sentiment signals | daily_score, sentiment_5dma, sentiment_momentum |
| Action | 3 multipliers ∈ [0.5, 1.5] — one per non-cash class |
| Reward | Return delta vs Agent 1 baseline |
| Timesteps | 300,000 |

### Agent 3 — SAC Stock Picker

| Property | Value |
|----------|-------|
| Algorithm | SAC (Stable Baselines3) |
| State dim | 856 = Z(256) + 50 stocks × 12 features |
| Per-stock features | EMA5, EMA20, RSI14, MACD, MACD_hist, ATR14, BB_width, OBV, MFI14, daily_return, log_return, rolling_sharpe_30 |
| Action | 50 stock weights → softmax |
| Reward | Sharpe − (0.1 × mean pairwise correlation) |
| Timesteps | 500,000 |

---

## Coordinator (No Neural Net)

```python
final_weight_i = class_weight_i × sentiment_modifier_i × stock_weight_i
# softmax normalize → all weights sum to 1
```

---

## Training Order (Critical)

```
1. Pretrain Transformer (P6) ✅ DONE — val_loss=0.000098, 46 features, epoch 20
2. Train Agent 1 (SAC) until Calmar stabilizes → freeze
3. Train Agent 3 (SAC) in parallel → freeze
4. Train Agent 2 (PPO) on top of frozen A1 + A3
5. Fine-tune all three jointly with LR = 1e-5
```

---

## Backtesting Baselines (6 Required)

1. PPO flat — single PPO, OHLCV only
2. SAC flat — single SAC, OHLCV only
3. A2C — standard, OHLCV
4. Markowitz — mean-variance optimization
5. Buy-and-hold NIFTY 50 index
6. Equal weight (2% per stock)

**Metrics:** Sharpe, Max Drawdown, CAGR, Calmar, Sortino (via `quantstats`)

---

## Stock Universe Note
Agent 3 operates on 50 stocks. TATAMOTORS.NS is 100% missing (yfinance 404)
and was excluded. The npy arrays use 49 effective tickers but the architecture
dimension (856-d) was computed as 50 stocks × 12 = 600.
⚠️ UNRESOLVED: Confirm from stock_features_meta.pkl whether padding was used
or actual dimension is 844 (49 × 12 + 256). Must verify before Agent 3 training.
```

***

Replace your entire `01_LOCKED_ARCHITECTURE.md` with this. Then we're clean for P9.