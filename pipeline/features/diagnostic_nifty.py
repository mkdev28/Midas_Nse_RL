# pipeline/features/diagnosis_nifty.py
from datasets import load_dataset
import pandas as pd

ds = load_dataset("raeidsaqur/NIFTY", split="train")
df = pd.DataFrame(ds)

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Dtypes:\n", df.dtypes)
print("\nHead (3 rows):")
print(df.head(3).to_string())
print("\n── Date column ──")
print("Unique date sample:", df["date"].dropna().head(10).tolist() if "date" in df.columns else "NO DATE COLUMN")
print("Date range:", pd.to_datetime(df["date"], errors="coerce").min(), "→",
      pd.to_datetime(df["date"], errors="coerce").max())
print("Null dates:", df["date"].isna().sum() if "date" in df.columns else "N/A")

print("\n── Text/News column ──")
# Check all columns for news-like content
for col in df.columns:
    sample = str(df[col].iloc[0])
    print(f"  [{col}] first 200 chars: {sample[:200]}")

print("\n── Label check ──")
for col in df.columns:
    if df[col].dtype == object:
        uniq = df[col].nunique()
        if uniq < 20:
            print(f"  [{col}] unique values ({uniq}): {df[col].unique().tolist()}")