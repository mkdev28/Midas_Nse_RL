import pandas as pd
df = pd.read_parquet("data/processed/train.parquet")
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}):")
for i, c in enumerate(df.columns):
    print(f"  {i+1:02d}. {c}")