from datasets import load_dataset
import pandas as pd
import os

os.makedirs("raw", exist_ok=True)

# Option 1 — Correct NIFTY dataset ID (US financial headlines, usable for FinBERT domain adaptation)
print("Trying raeidsaqur/NIFTY...")
try:
    dataset = load_dataset("raeidsaqur/NIFTY", trust_remote_code=True)
    df1 = dataset['train'].to_pandas()
    print(f"raeidsaqur/NIFTY → Shape: {df1.shape}, Columns: {df1.columns.tolist()}")
    df1.to_csv("raw/news_nifty_huggingface.csv", index=False)
    print("Saved ✅")
except Exception as e:
    print(f"Failed: {e}")

# Option 2 — Indian Financial News (26k rows, NSE-specific) ✅ Better for our use case
print("\nTrying kdave/Indian_Financial_News...")
try:
    dataset2 = load_dataset("kdave/Indian_Financial_News")
    df2 = dataset2['train'].to_pandas()
    print(f"kdave/Indian_Financial_News → Shape: {df2.shape}, Columns: {df2.columns.tolist()}")
    print(df2.head(3))
    df2.to_csv("raw/news_india_financial.csv", index=False)
    print("Saved ✅")
except Exception as e:
    print(f"Failed: {e}")

# Option 3 — nickmuchi financial classification (FinBERT-compatible sentiment labels)
print("\nTrying nickmuchi/financial-classification...")
try:
    dataset3 = load_dataset("nickmuchi/financial-classification")
    df3 = dataset3['train'].to_pandas()
    print(f"nickmuchi/financial-classification → Shape: {df3.shape}, Columns: {df3.columns.tolist()}")
    print(df3.head(3))
    df3.to_csv("raw/news_financial_classification.csv", index=False)
    print("Saved ✅")
except Exception as e:
    print(f"Failed: {e}")