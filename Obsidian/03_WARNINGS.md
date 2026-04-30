# MIDAS-NSE — Warnings and Constraints

> **Purpose:** Every LLM must read this file before writing any code. These are hard constraints that prevent the most common mistakes new sessions introduce.
> **Last updated:** 2026-04-30

---

## ⛔ Critical Warnings

### W1 — Sentiment Data is Currently All Zeros
The sentiment columns in `features_v1.parquet`, `train.parquet`, `val.parquet`, and `test.parquet` are **placeholder zeros**. Do NOT proceed to P9 (Gym environment) or P10 (Agent training) until:
1. P7 produces `daily_sentiment.csv` with real FinBERT scores
2. P8 merges sentiment into the parquets
3. P6 Transformer is retrained on the updated data

### W2 — Transformer Input Features Must Match Gym Environment
The 36 features fed to the Transformer at pretrain time must be **exactly the same 36 features** the Gym environment feeds during inference. Verify by loading `transformer_best.pt` and checking `checkpoint['feature_cols']` before building P9.

### W3 — Scaler Must Be Refit After Sentiment Merge
`scaler.pkl` was fitted on `features_v1` with zero sentiment. After P8 updates sentiment columns with real values, the scaler **must be refit on the new train.parquet**. The old `scaler.pkl` becomes invalid.

### W4 — 50 Stocks, Not 15 (CORRECTED)
Agent 3 operates on all 50 NIFTY stocks. The tensor shape [T, 50, 12] is final.
- MM.NS: removed as duplicate Adj Close column artefact in P5
- TATAMOTORS.NS: 100% missing data (yfinance 404) — still occupies one of the 50 
  slots in stock_features.npy as all-NaN rows. Agent 3 trains on all 50 positions.
Do not reduce to 49 or any other number. The 50-slot architecture is locked.

### W5 — No Reddit Data
Agent 2 state uses only FinBERT sentiment signals + frozen Agent 1 weights. Do **not** add Reddit PRAW or any social media data source.

### W6 — Transformer is Frozen During RL Training
During P10 (agent training), only the three agent policies update. The encoder weights in `transformer_encoder.pt` do **not** change. Load the encoder with `model.eval()` and `torch.no_grad()`.

### W7 — 26-Row Offset Between Parquets and Stock Feature Arrays
train.parquet has 3153 rows. X_train_technical.npy has 3179 rows.
The 26-row difference comes from features_v1.parquet dropping warmup NaNs 
(RSI/MACD warmup) that stock_features.npy did not drop.

BEFORE building P9, run:
  pd.read_parquet("data/processed/train.parquet").index[0]
  # and compare against stock_features_meta.pkl start_date

The Gym environment must index both arrays from the same start date.
Using positional index alone will cause a 26-day misalignment in every lookup.
Discovered: 2026-04-30

### W8 — Fine-tune Own FinBERT, Do Not Use Pre-built Variants
Decision (2026-04-30): Fine-tune ProsusAI/finbert on kdave/Indian_Financial_News 
(26k GPT-labeled rows). Do NOT substitute pre-built variants like 
finbert-sentfin or Vansh180/FinBERT-India-v1.
Reason: Our corpus is larger, India-specific, and gives strictly better domain fit.
Fine-tune batch_size=32, inference batch_size=64.
First run the P7.1 diagnostic before writing any P7 code.
Discovered: 2026-04-30

### W9 — Encoder Checkpoint is a Nested Dict (Not Raw state_dict)
`transformer_encoder.pt` was saved as `{"encoder_state": ..., "feature_cols": ..., "config": ...}`.
To load: `ckpt["encoder_state"]` → pass to `model.load_state_dict()`.
Do NOT call `load_state_dict(ckpt)` directly — will crash with missing key errors.
Discovered: 2026-04-30

### W10 — Encoder Input Dim is 46, Not 36
The transformer encoder was trained with 46 features (full train.parquet columns),
not the 36 stated in the locked architecture doc.
`_MidasEncoder.input_proj` must be `Linear(46, 256)`. `pos_enc max_len=500`.
Discovered: 2026-04-30

### W11 — Column Name Corrections in train.parquet
Actual column names differ from architecture doc:
- `inr_usd_close` → `inrusd_close`
- `fii_net_cr` → `fii_net`
Verify all column references against `pd.read_parquet("data/processed/train.parquet").columns`.
Discovered: 2026-04-30
---

## 🔧 Technical Constraints

### TC1 — Environment
- **OS:** Windows only, PowerShell
- **Python:** 3.12
- **GPU:** RTX 4060 8GB VRAM — batch sizes must respect this
- **RAM:** 16GB — large DataFrames may need chunking

### TC2 — No API Calls During Training
All data must be pre-computed and saved to parquet/npy **before** the Gym environment is built. `env.step()` must only do DataFrame/array lookups. Zero yfinance, zero FinBERT, zero scraping during training.

### TC3 — Config Paths
```python
RAW  = "data/raw"          # raw downloaded files
PROC = "data/processed"    # cleaned, engineered, normalized outputs
CKPT = "checkpoints"       # model checkpoints
```
All paths are relative to project root: `C:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse\`

### TC4 — Training Order is Sequential
```
P6 (Transformer) → A1 (SAC) → A3 (SAC, parallel OK) → A2 (PPO) → Joint fine-tune
```
Agent 2 cannot be trained until Agent 1 is frozen. Do not train them simultaneously.

---

## 📝 How to Add New Warnings

When a session reveals a new constraint, gotcha, or mistake-prone area, add it here with:
- A unique ID (W7, W8... or TC5, TC6...)
- A clear one-line summary
- The consequence of ignoring it
- The date it was discovered

---

## Changelog

| Date | Warning | Added by |
|------|---------|----------|
| 2026-04-30 | W1-W6, TC1-TC4 | Initial setup from handoff document |
