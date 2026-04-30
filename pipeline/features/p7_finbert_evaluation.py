# pipeline/features/p7_evaluate_finbert.py

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from pathlib import Path

CKPT   = Path("checkpoints")
FINETUNED_MODEL = CKPT / "finbert_india"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL2ID = {"Positive": 0, "Negative": 1, "Neutral": 2}
ID2LABEL = {0: "Positive", 1: "Negative", 2: "Neutral"}

print(f"Device: {DEVICE}")

# ── Load val split (same seed as training) ────────────────────────────────────
ds = load_dataset("kdave/Indian_Financial_News", split="train")
df = pd.DataFrame(ds)
df = df[df["Summary"].str.strip().str.len() > 20].copy()
df["label_id"] = df["Sentiment"].map(LABEL2ID)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

_, val_df = train_test_split(df, test_size=0.1, random_state=42,
                              stratify=df["label_id"])
print(f"Val set: {len(val_df)} rows")
print(f"Val label distribution:\n{val_df['Sentiment'].value_counts()}")

# ── Load fine-tuned model ─────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_MODEL))
model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_MODEL))
model.to(DEVICE)
model.eval()

# ── Run inference on val set ──────────────────────────────────────────────────
all_preds, all_probs, all_labels = [], [], []
BATCH = 64

texts  = val_df["Summary"].tolist()
labels = val_df["label_id"].tolist()

for i in range(0, len(texts), BATCH):
    batch_texts = texts[i:i+BATCH]
    enc = tokenizer(batch_texts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = softmax(model(**enc).logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    all_preds.extend(preds)
    all_probs.extend(probs)
    all_labels.extend(labels[i:i+BATCH])

all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n══ CLASSIFICATION REPORT ══════════════════════════════")
print(classification_report(
    all_labels, all_preds,
    target_names=["Positive", "Negative", "Neutral"]
))

print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"Macro F1:         {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"Weighted F1:      {f1_score(all_labels, all_preds, average='weighted'):.4f}")

print("\n══ CONFUSION MATRIX ════════════════════════════════════")
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm,
    index=["True_Pos","True_Neg","True_Neu"],
    columns=["Pred_Pos","Pred_Neg","Pred_Neu"])
print(cm_df)
print("\nMost common misclassification pairs:")
for i in range(3):
    for j in range(3):
        if i != j and cm[i,j] > 0:
            print(f"  {ID2LABEL[i]} → predicted as {ID2LABEL[j]}: {cm[i,j]} times")

# ── Calibration — confidence vs accuracy by decile ───────────────────────────
print("\n══ CALIBRATION CHECK ═══════════════════════════════════")
max_probs   = all_probs.max(axis=1)
correct     = (all_preds == all_labels).astype(int)
cal_df = pd.DataFrame({"confidence": max_probs, "correct": correct})
cal_df["decile"] = pd.cut(cal_df["confidence"], bins=10, precision=2)
cal_summary = cal_df.groupby("decile", observed=True).agg(
    count=("correct","count"),
    accuracy=("correct","mean"),
    avg_conf=("confidence","mean")
).reset_index()
print(cal_summary.to_string())
print("\nWell-calibrated = accuracy ≈ avg_conf in each row")

# ── Domain sanity test — India-specific headlines ────────────────────────────
print("\n══ DOMAIN SANITY TEST ══════════════════════════════════")
test_headlines = [
    ("Sensex crashes 1500 points as FII selloff intensifies",     "Negative"),
    ("Nifty hits all-time high, Sensex crosses 80000 mark",       "Positive"),
    ("RBI holds repo rate steady at 6.5 percent",                 "Neutral"),
    ("India GDP growth slows to 5.4 percent, below estimates",    "Negative"),
    ("Adani Group stocks rally 8 percent on strong Q3 results",   "Positive"),
    ("SEBI issues new circular on F&O margin requirements",       "Neutral"),
    ("FII outflows hit 3-month high amid global risk-off mood",   "Negative"),
    ("Mutual fund SIP inflows cross Rs 20000 crore for first time", "Positive"),
]

enc = tokenizer([h for h,_ in test_headlines], padding=True, truncation=True,
                max_length=128, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    probs = softmax(model(**enc).logits, dim=1).cpu().numpy()

print(f"{'Headline':<55} {'Expected':<10} {'Predicted':<10} {'Conf':>6} {'✓'}")
print("─"*90)
for i, (headline, expected) in enumerate(test_headlines):
    pred_id   = probs[i].argmax()
    predicted = ID2LABEL[pred_id]
    confidence= probs[i].max()
    correct   = "✅" if predicted == expected else "❌"
    print(f"{headline[:54]:<55} {expected:<10} {predicted:<10} {confidence:>6.2%} {correct}")