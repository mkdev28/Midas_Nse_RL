# pipeline/features/p7_finbert_sentiment.py
"""
P7 — Three-Tier Sentiment Construction
──────────────────────────────────────────────────────────────────────────────
Tier 1  2008-2013  India VIX pseudo-sentiment   sentiment_available=0
Tier 2  2014-2024  FinBERT on RSS (if available) sentiment_available=1
                   VIX proxy as fallback on gap days
Tier 3  2024-2025  FinBERT on RSS               sentiment_available=1

Fine-tune corpus: kdave/Indian_Financial_News (26k GPT-labeled, India-specific)
Inference target: ET RSS + MoneyControl RSS (dated headlines)
VIX formula:      clip(-(vix_close - vix_20dma) / vix_20dma, -1, 1)
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from datasets import load_dataset
import feedparser
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
RAW              = Path("data/raw")
PROC             = Path("data/processed")
CKPT             = Path("checkpoints")
BASE_MODEL       = "ProsusAI/finbert"        # W8: fine-tune this, NEVER pre-built variants
FINETUNED_MODEL  = CKPT / "finbert_india"
BATCH_SIZE_TRAIN = 32                        # TC1: RTX 4060 8GB
BATCH_SIZE_INFER = 64
CHECKPOINT_FILE  = PROC / "headline_sentiment_checkpoint.csv"
SENTIMENT_START  = pd.Timestamp("2014-01-01")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# ProsusAI/finbert native label order — locked, do not change
LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

print(f"Device: {DEVICE}")
PROC.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# P7.2 — Fine-tune ProsusAI/finbert on kdave
# ══════════════════════════════════════════════════════════════════════════════
print("\n── P7.2: Fine-tuning ProsusAI/finbert on kdave ──")

# Skip fine-tuning if model already exists (crash recovery)
if (FINETUNED_MODEL / "config.json").exists():
    print(f"Fine-tuned model already exists at {FINETUNED_MODEL} — skipping fine-tune.")
else:
    ds = load_dataset("kdave/Indian_Financial_News", split="train")
    df_kdave = pd.DataFrame(ds)
    df_kdave = df_kdave[df_kdave["Summary"].str.strip().str.len() > 20].copy()
    df_kdave["label_id"] = df_kdave["Sentiment"].map(LABEL2ID)
    df_kdave = df_kdave.dropna(subset=["label_id"])
    df_kdave["label_id"] = df_kdave["label_id"].astype(int)

    print(f"kdave rows after cleaning: {len(df_kdave)}")
    print(f"Label distribution:\n{df_kdave['Sentiment'].value_counts()}")

    train_df, val_df = train_test_split(
        df_kdave, test_size=0.1, random_state=42, stratify=df_kdave["label_id"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len=128):
            self.enc = tok(list(texts), truncation=True, padding=True,
                           max_length=max_len, return_tensors=None)
            self.labels = list(labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = SentimentDataset(train_df["Summary"], train_df["label_id"], tokenizer)
    val_dataset   = SentimentDataset(val_df["Summary"],   val_df["label_id"],   tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()}
    )
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=str(FINETUNED_MODEL),
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_TRAIN,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),      # RTX 4060 supports fp16 — ~2x speedup
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,            # TC1: Windows/PowerShell
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
    )
    print("Starting fine-tuning (~15-20 min on RTX 4060)...")
    trainer.train()
    trainer.save_model(str(FINETUNED_MODEL))
    tokenizer.save_pretrained(str(FINETUNED_MODEL))
    print(f"✅ Fine-tuned model saved → {FINETUNED_MODEL}")


# ══════════════════════════════════════════════════════════════════════════════
# P7.3 — Fetch RSS headlines (MoneyControl + ET)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── P7.3: Fetching RSS headlines ──")

rss_rows = []
feeds = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketsnews.xml",
    "et_markets":   "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
}
for source, url in feeds.items():
    feed = feedparser.parse(url)
    for entry in feed.entries:
        raw_date = entry.get("published", "")
        parsed_date = pd.to_datetime(raw_date, errors="coerce", utc=True)
        rss_rows.append({
            "date":   parsed_date,
            "text":   str(entry.get("title", ""))[:512],
            "source": source,
        })
    print(f"  {source}: {len(feed.entries)} entries")

df_rss = pd.DataFrame(rss_rows)
if not df_rss.empty and pd.api.types.is_datetime64_any_dtype(df_rss["date"]):
    df_rss["date"] = df_rss["date"].dt.tz_localize(None).dt.normalize()
df_rss = df_rss[df_rss["text"].str.len() > 15].reset_index(drop=True)
df_rss.to_csv(RAW / "rss_headlines.csv", index=False)
print(f"  Total RSS headlines: {len(df_rss)}")
print(f"  Date range: {df_rss['date'].min()} → {df_rss['date'].max()}")


# ══════════════════════════════════════════════════════════════════════════════
# P7.4 — Inference on RSS with fine-tuned model (crash-safe)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── P7.4: Running inference on RSS headlines ──")

inf_tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_MODEL))
inf_model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_MODEL))
inf_model.to(DEVICE)
inf_model.eval()
print(f"Loaded fine-tuned model | labels: {inf_model.config.id2label}")

# Crash recovery — resume from last checkpoint
if CHECKPOINT_FILE.exists():
    df_done = pd.read_csv(CHECKPOINT_FILE)
    start_idx = len(df_done)
    results = df_done.to_dict("records")
    print(f"Resuming from checkpoint at row {start_idx}")
else:
    results = []
    start_idx = 0

remaining = df_rss.iloc[start_idx:].reset_index(drop=True)
print(f"Running inference on {len(remaining)} headlines...")

for batch_start in range(0, len(remaining), BATCH_SIZE_INFER):
    batch = remaining.iloc[batch_start: batch_start + BATCH_SIZE_INFER]
    encoded = inf_tokenizer(
        batch["text"].tolist(), padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        probs = softmax(inf_model(**encoded).logits, dim=1).cpu().numpy()

    # ID2LABEL = {0: positive, 1: negative, 2: neutral} — locked order
    for i, row in enumerate(batch.itertuples(index=False)):
        results.append({
            "date":   row.date,
            "text":   row.text,
            "source": row.source,
            "pos":    float(probs[i, 0]),
            "neg":    float(probs[i, 1]),
            "neu":    float(probs[i, 2]),
        })

    completed = start_idx + batch_start + len(batch)
    if (completed // 500) > ((completed - len(batch)) // 500):
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
        print(f"  Checkpoint saved at {completed}/{len(df_rss)}")

df_headline_sentiment = pd.DataFrame(results)
df_headline_sentiment.to_csv(PROC / "headline_sentiment.csv", index=False)
print(f"✅ headline_sentiment.csv saved: {len(df_headline_sentiment)} rows")


# ══════════════════════════════════════════════════════════════════════════════
# P7.5 — Build daily_sentiment.csv (three-tier)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── P7.5: Building daily_sentiment.csv ──")

# ── Load trading calendar as spine ───────────────────────────────────────────
cal = pd.read_parquet(PROC / "trading_calendar.parquet")
cal["date"] = pd.to_datetime(cal["date"])
full = cal[["date"]].copy().sort_values("date").reset_index(drop=True)

# ── Compute VIX pseudo-sentiment for ALL calendar days ───────────────────────
# Formula: clip(-(vix_close - vix_20dma) / vix_20dma, -1, 1)
# When VIX spikes above 20dma → negative sentiment. Falls below → positive.
vix_raw = pd.read_parquet(RAW / "india_vix.parquet").reset_index()

# Handle yfinance index structure robustly
date_candidates = [c for c in vix_raw.columns
                   if str(c).lower() in ("date", "datetime", "index")]
if date_candidates:
    vix_raw = vix_raw.rename(columns={date_candidates[0]: "date"})
else:
    vix_raw = vix_raw.rename(columns={vix_raw.columns[0]: "date"})

# Get Close column (yfinance: Open/High/Low/Close/Adj Close/Volume)
num_cols = vix_raw.select_dtypes(include=np.number).columns.tolist()
close_col = next((c for c in num_cols if "close" in str(c).lower()
                  and "adj" not in str(c).lower()), num_cols[0])
vix = (vix_raw[["date", close_col]]
       .rename(columns={close_col: "vix_close"})
       .copy())
vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
vix = vix.sort_values("date").drop_duplicates("date").reset_index(drop=True)
vix["vix_20dma"] = vix["vix_close"].rolling(20, min_periods=5).mean()
vix["vix_pseudo"] = np.clip(
    -(vix["vix_close"] - vix["vix_20dma"]) / vix["vix_20dma"],
    -1.0, 1.0
)
print(f"VIX data: {len(vix)} rows | {vix['date'].min()} → {vix['date'].max()}")
print(f"VIX pseudo range: {vix['vix_pseudo'].min():.4f} → {vix['vix_pseudo'].max():.4f}")

full = full.merge(vix[["date", "vix_pseudo"]], on="date", how="left")
full["vix_pseudo"] = full["vix_pseudo"].fillna(0)   # pre-VIX data days → 0

# ── Aggregate FinBERT scores from RSS headlines ───────────────────────────────
df_dated = df_headline_sentiment.dropna(subset=["date"]).copy()
df_dated["date"] = pd.to_datetime(df_dated["date"]).dt.normalize()
df_dated["sentiment_score"] = df_dated["pos"] - df_dated["neg"]

def get_dominant(row):
    if row["pos"] >= row["neg"] and row["pos"] >= row["neu"]:
        return "pos"
    elif row["neg"] >= row["neu"]:
        return "neg"
    return "neu"

df_dated = df_dated.copy()
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

# ── Combine tiers into daily_score ────────────────────────────────────────────
# Days with FinBERT scores → use FinBERT
# Days without (gap periods) → use VIX proxy
full["has_finbert"]     = full["finbert_score"].notna()
full["daily_score"]     = np.where(
    full["has_finbert"], full["finbert_score"], full["vix_pseudo"]
)
full["daily_pos_count"] = full["daily_pos_count"].fillna(0).astype(int)
full["daily_neg_count"] = full["daily_neg_count"].fillna(0).astype(int)

# ── sentiment_available flag ──────────────────────────────────────────────────
# 0 → pre-2014 (VIX proxy only, no textual data available)
# 1 → 2014+    (FinBERT where headlines exist, VIX proxy as fallback — Agent 2 uses this)
full["sentiment_available"] = (full["date"] >= SENTIMENT_START).astype(int)

# Hard enforce pre-2014 zeros
mask_pre2014 = full["date"] < SENTIMENT_START
full.loc[mask_pre2014, "daily_score"]      = full.loc[mask_pre2014, "vix_pseudo"]
full.loc[mask_pre2014, "daily_pos_count"]  = 0
full.loc[mask_pre2014, "daily_neg_count"]  = 0
full.loc[mask_pre2014, "sentiment_available"] = 0

# ── Rolling features ──────────────────────────────────────────────────────────
full = full.sort_values("date").reset_index(drop=True)
full["sentiment_5dma"]     = full["daily_score"].rolling(5, min_periods=1).mean()
full["sentiment_vol"]      = full["daily_score"].rolling(5, min_periods=2).std().fillna(0)
full["sentiment_momentum"] = full["daily_score"] - full["sentiment_5dma"]

# ── Final output — exactly 8 required columns ─────────────────────────────────
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
      [["date","daily_score","sentiment_available"]].head(8).to_string())

print(f"\nSanity — 2020 COVID crash (expect negative):")
print(full[(full["date"]>="2020-02-15") & (full["date"]<="2020-04-15")]
      [["date","daily_score","sentiment_available"]].head(8).to_string())

print(f"\nSanity — recent RSS days (FinBERT, expect pos_count > 0 on some):")
print(full[full["sentiment_available"]==1].tail(10)
      [["date","daily_score","daily_pos_count","daily_neg_count","sentiment_available"]].to_string())

full.to_csv(PROC / "daily_sentiment.csv", index=False)
print(f"\n✅ daily_sentiment.csv saved → {PROC / 'daily_sentiment.csv'}")
print(f"   Shape: {full.shape}")
print(f"   Columns: {full.columns.tolist()}")