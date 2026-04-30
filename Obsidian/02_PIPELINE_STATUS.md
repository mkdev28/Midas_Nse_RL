# MIDAS-NSE — Pipeline Status

> **Purpose:** Exact state of every pipeline phase. Update this after every work session.
> Last updated: 2026-04-30 (P9 completed) | Updated by: Mohit
---

## Data Downloads

| ID | Dataset | File | Shape | Status |
|----|---------|------|-------|--------|
| D1 | NIFTY 50 OHLCV | `data/raw/nifty50_ohlcv.parquet` | [4438, 257] MultiIndex | ✅ Done |
| D2 | NIFTY 50 Index | `data/raw/nifty_index.parquet` | [4414, 6] | ✅ Done |
| D3 | India VIX | `data/raw/india_vix.parquet` | [4368, 6] starts 2008-03-03 | ✅ Done |
| D4 | Gold futures | `data/raw/gold.parquet` | [4527, 6] | ✅ Done |
| D5 | Crude futures | `data/raw/crude.parquet` | [4528, 6] | ✅ Done |
| D6 | INR/USD | `data/raw/inr_usd.parquet` | [4666, 6] | ✅ Done |
| D7 | FII/DII flows | `data/raw/fii_dii_flows.csv` | ~4400 rows | ✅ Done |
| D8 | RBI repo rate | `data/raw/rbi_repo_rate_raw.csv` | manual RBI DBIE | ✅ Done |
| D9 | G-Sec 10Y yield | `data/raw/gsec_yield_raw.csv` | manual RBI DBIE | ✅ Done |
D10 | News headlines | data/external/news_india_financial.csv | [26961, 4] kdave corpus | ✅ Done
D11 | MoneyControl RSS | data/external/news_moneycontrol_rss.csv | 0 rows — feed dead | ✅ Done (empty)
D12 | ET RSS | data/external/news_economic_times_rss.csv | [50, 4] live only | ✅ Done
### Known Data Issues (Accepted)

- **TATAMOTORS.NS:** 100% missing — yfinance 404, accepted and logged
- **COALINDIA.NS:** 15.8% missing — IPO Nov 2010, expected
- **SBILIFE.NS:** 54.1% missing — IPO 2017, expected
- **HDFCLIFE.NS:** 54.82% missing — IPO 2017, expected
- **MM.NS:** Extra ticker in Adj Close column — removed in P5
- **India VIX Jan-Feb 2008 gap:** Forward-filled from March 2 value
- **Crude April 2020 negative price:** Retained as real data, `extreme_event` flag set

---

## Pipeline Phases

### ✅ P1 — Master Alignment
- **Script:** `pipeline/features/p1_calendar.py`
- **Output:** `data/processed/trading_calendar.parquet`, `data/processed/master_aligned.parquet`
- **Shape:** 4414 rows × 25 columns
- **Method:** NSE trading calendar as spine. All datasets aligned via left-join. Forward-fill on repo rate, G-Sec, VIX gap.

### ✅ P2 — Cleaning
- **Script:** `pipeline/features/p2_align.py`
- **Output:** Cleaned `master_aligned.parquet`
- **Method:** Impossible OHLCV values checked, circuit breakers flagged, extreme moves logged.

### ✅ P3 — Feature Engineering (Index-level)
- **Script:** `pipeline/features/p3_features.py`
- **Output:** `data/processed/features_v1.parquet`
-Shape: 4387 rows × 48 columns (patched 2026-04-30 — added 3 C1 features)
Added features: dii_absorption_ratio, vix_regime, institutional_net
- **Features:** NIFTY returns (1d, 5d, 20d), RSI14, MACD, Bollinger Bands, ATR, volume ratio, yield_spread, gold_crude_ratio, FII/DII momentum (5d, 20d rolling), `y_next_day_return` target
- **Note:** Warmup NaNs for first 26 rows (expected). Last row dropped (y leakage prevention).

### ✅ P4 — Split and Normalize
- **Script:** `pipeline/features/p4_split_normalize.py`
- **Output:** `data/processed/train.parquet`, `val.parquet`, `test.parquet`, `scaler.pkl`

| Split | Rows | Period |
|-------|------|--------|
| Train | 3153 | 2008-02-06 to 2020-12-31 |
| Val | 496 | 2021-01-01 to 2022-12-30 |
| Test | 738 | 2023-01-02 to 2025-12-29 |

- **Scaler:** RobustScaler fitted on train only. Train mean=0.000000, std=1.000000. Zero nulls.

### ✅ P5 — Per-stock Technical Features
- **Script:** `pipeline/features/p5_stock_features.py`
- **Output:** `data/processed/stock_features.npy`, `X_train_technical.npy`, `X_val_technical.npy`, `X_test_technical.npy`, `stock_features_meta.pkl`

| Array | Shape |
|-------|-------|
| stock_features.npy (full) | [4414, 50, 12] |
| X_train_technical | [3179, 50, 12] |
| X_val_technical | [496, 50, 12] |
| X_test_technical | [739, 50, 12] |

- **Stocks:** 50 tickers (MM.NS removed)
- **NaN:** 1.667% (IPO gaps + warmup)
- **Train mean:** 0.0000, std: 0.9487
- **Feature order:** EMA5, EMA20, RSI14, MACD, MACD_hist, ATR14, BB_width, OBV, MFI14, daily_return, log_return, rolling_sharpe_30

⚠️ ALIGNMENT WARNING: X_train_technical has 3179 rows vs train.parquet's 3153 rows.
26-row offset exists due to different warmup handling. Must resolve before P9 by
confirming which trading dates each array starts on and aligning to the same spine.
Run: pd.read_parquet("data/processed/train.parquet").index[0] vs
     stock_features_meta.pkl start_date before building Gym env.

### ✅ P6 — Transformer Pretrain
- Script: pipeline/features/p6_transformer_pretrain.py
- Output: checkpoints/transformer_best.pt, transformer_encoder.pt, pretrain_history.csv
- Best val loss: 0.000098 at epoch 20
- Early stopped: Epoch 35
- Features: 46 (verified via checkpoint['feature_cols'])
- Z dimension: 256
- Retrained: 2026-04-30 with real sentiment + 3 C1 features

### ✅ P7 — FinBERT Sentiment
- Script: pipeline/features/p7_finbert_sentiment.py
- Output: data/processed/daily_sentiment.csv, data/processed/headline_sentiment.csv
- Model: checkpoints/finbert_india/ (fine-tuned ProsusAI/finbert on kdave)
- Fine-tune metrics: Val accuracy 91.76%, Macro F1=0.917, 8/8 domain test pass
- daily_sentiment.csv: 4414 rows, 8 cols, zero nulls
- Sentiment coverage: 2014-01-01 onwards (sentiment_available=1)
                      Pre-2014: VIX pseudo-sentiment proxy (sentiment_available=0)
- RSS: 50 ET headlines dated 2026-04-30 (outside training window, not merged)

### ✅ P8 — Merge Sentiment into Parquets
- Script: pipeline/features/p8_merge_sentiment.py
- Merged daily_sentiment.csv into all three parquets
- Scaler refit on train only (W3 resolved)
- Train: (3153, 54) | Val: (496, 54) | Test: (738, 54)
- All splits: zero nulls
- sentiment_available=1: val=496/496, test=738/738, train=1712/3153



### ✅ P9 — Gym Environment
- **Script:** `pipeline/packaging/p9_gym_env.py`
- **Status:** Smoke test passed 2026-04-30
- **Agent obs/act dims confirmed:**
  - A1: obs=(265,) act=(4,)
  - A3: obs=(856,) act=(50,)
  - A2: obs=(7,) act=(3,)
- **Coordinator:** final_weights sum=1.0 ✅
- **Replay buffer:** normal + tail-risk buckets verified ✅
- **Fixes applied during P9:**
  - Column names: `inr_usd_close` → `inrusd_close`, `fii_net_cr` → `fii_net`
  - Encoder saved as nested dict — keys: `encoder_state`, `feature_cols`, `config`
  - Encoder input dim is 46 (not 36 — trained with full parquet feature set)
  - `transformer_best.pt` requires `weights_only=False` (PyTorch 2.6)

### ⬜ P10 — Agent Training
- **Goal:** Train A1 → A3 → A2 → joint fine-tune. SB3.

### ⬜ P11 — Backtesting
- **Goal:** Test period 2023-2025 (738 days). 6 baselines. quantstats metrics.

### ⬜ P12 — XAI
- **Goal:** SHAP on A1/A3 + Transformer attention weights. Figures for paper.

---

## Files on Disk

### `data/raw/`
```
nifty50_ohlcv.parquet      [4438, 257] MultiIndex
nifty_index.parquet        [4414, 6]
india_vix.parquet          [4368, 6]
gold.parquet               [4527, 6]
crude.parquet              [4528, 6]
inr_usd.parquet            [4666, 6]
fii_dii_flows.csv          ~4400 rows
rbi_repo_rate_raw.csv      manual download
gsec_yield_raw.csv         manual download
```

### `data/processed/`
```
trading_calendar.parquet   4414 NSE trading days
master_aligned.parquet     [4414, 27] — includes 3 C1 features
features_v1.parquet        [4387, 48] — includes 3 C1 features
train.parquet              [3153, 54] — real sentiment merged
val.parquet                [496, 54]  — real sentiment merged
test.parquet               [738, 54]  — real sentiment merged
scaler.pkl                 RobustScaler refit on 50-feature train
daily_sentiment.csv        [4414, 8]  — FinBERT + VIX proxy
headline_sentiment.csv     [50, n]    — ET RSS Apr 2026 inference
stock_features.npy         [4414, 50, 12]
X_train_technical.npy      [3179, 50, 12]  ⚠️ 26-row offset vs train.parquet
X_val_technical.npy        [496, 50, 12]
X_test_technical.npy       [739, 50, 12]
stock_features_meta.pkl    ticker list + feature names
```

### `checkpoints/`
```
transformer_best.pt        46 features, epoch 20, val_loss=0.000098
transformer_encoder.pt     encoder only — what RL agents load
pretrain_history.csv       35 rows epoch/train_loss/val_loss
finbert_india/             fine-tuned FinBERT — F1=0.917
```
