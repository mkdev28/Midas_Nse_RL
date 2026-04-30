# MIDAS-NSE — Next Steps

> **Purpose:** Always contains exactly what to do next. The first thing any new LLM session reads after the warnings.
> **Last updated:** 2026-04-30

---

## 🎯 Current Task: P7 — FinBERT Sentiment

**Goal:** Produce `data/processed/daily_sentiment.csv` with 7 daily sentiment features covering 2008-2025.

**Required output columns:**
```
date, daily_score, daily_pos_count, daily_neg_count,
sentiment_5dma, sentiment_vol, sentiment_momentum, sentiment_available
```

Pre-2014 rows use neutral values with `sentiment_available=0`.

---

### Step P7.1 — Diagnostic (run FIRST before writing any code)

```python
from datasets import load_dataset
import pandas as pd
ds = load_dataset("kdave/Indian_Financial_News", split="train")
df = pd.DataFrame(ds)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Dtypes:", df.dtypes)
print(df.head(3).to_string())
```

> ⚠️ Paste the output back to the LLM. The P7 script structure depends on what columns and labels exist in this dataset.

---

### Step P7.2 — Fetch RSS headlines

```python
import feedparser, pandas as pd
mc = feedparser.parse("https://www.moneycontrol.com/rss/marketsnews.xml")
et = feedparser.parse("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")
rows = []
for entry in mc.entries + et.entries:
    rows.append({'date': entry.get('published',''), 'headline': entry.get('title','')})
pd.DataFrame(rows).to_csv("data/raw/rss_headlines.csv", index=False)
```

---

### Step P7.3 — Fine-tune FinBERT on Indian headlines

- Load `ProsusAI/finbert` from HuggingFace
- If `kdave` dataset has sentiment labels → fine-tune directly
- If no labels → create pseudo-labels via keyword matching:
  - **Negative:** crash, fall, loss, selloff, decline, slump, plunge, tumble, weak
  - **Positive:** rally, surge, profit, growth, gain, jump, rise, soar, strong, bullish
  - **Neutral:** everything else
- Fine-tune on GPU (RTX 4060) with `batch_size=32`
- Save fine-tuned model to `checkpoints/finbert_india/`

### Decision already made (2026-04-30):
kdave/Indian_Financial_News HAS sentiment labels (GPT-labeled positive/negative/neutral).
Fine-tune directly. Do NOT use pseudo-labels. Do NOT use pre-built variants.
Run P7.1 diagnostic only to confirm column names before writing the training script.
---

### Step P7.4 — Run inference on all headlines

- Batch size: 64
- Save per-headline probabilities to `data/processed/headline_sentiment.csv` as checkpoint
- **Save progress every 500 headlines** (crash recovery)

---

### Step P7.5 — Aggregate to daily level

- Group by date
- Compute all 7 features
- Set `sentiment_available=0` for all rows before 2014-01-01
- Save to `data/processed/daily_sentiment.csv`
- **Expected:** ~4500 rows. `daily_score` range ~[-0.5, +0.5]. Noticeably more negative in 2008, 2020 Q1, 2022.

---

## After P7 Completes → P8

1. Load `daily_sentiment.csv`
2. Load `train/val/test.parquet`
3. Drop existing zero sentiment columns
4. Left-join sentiment on date
5. Refit RobustScaler on train only
6. Re-normalize all three splits
7. Overwrite `train/val/test.parquet` and `scaler.pkl`
8. Verify: `sentiment_available=0` for pre-2014, `=1` after

---

## After P8 Completes → P6 Rerun

Retrain Transformer with same config, same script (`p6_transformer_pretrain.py`), just new data with real sentiment.

---
## P9 — Gym Environment

### ⚠️ Before writing any P9 code, run this first:
import pandas as pd, pickle
print(pd.read_parquet("data/processed/train.parquet").index[0])
with open("data/processed/stock_features_meta.pkl","rb") as f:
    meta = pickle.load(f)
print(meta)  # confirm start date and ticker list
# The two must start on the same trading date. See W7 in warnings.md.

## Full Remaining Roadmap

```
## Full Remaining Roadmap

P7  FinBERT Sentiment        ✅ DONE 2026-04-30
P8  Merge Sentiment          ✅ DONE 2026-04-30
P6  Retrain Transformer      ✅ DONE (P6 rerun after P8)
P9  Gym Environment          ✅ DONE 2026-04-30
P10 Agent Training           ← TEAMMATE STARTS HERE
P11 Backtesting
P12 XAI