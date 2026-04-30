import pandas as pd
import os

# Load both sources
kaggle = pd.read_parquet("raw/fii_dii_kaggle_clean.parquet")
kaggle['fii_dii_available'] = 1

trendlyne = pd.read_parquet("raw/fii_dii_trendlyne_2022_2026.parquet")
trendlyne['fii_dii_available'] = 1

# Kaggle ends Aug 2022, Trendlyne starts Jan 2022 — trim overlap
# Keep Kaggle up to where Trendlyne begins, then use Trendlyne
trendlyne_start = trendlyne['date'].min()
kaggle_trimmed = kaggle[kaggle['date'] < trendlyne_start]

print(f"Kaggle rows kept: {len(kaggle_trimmed)} (up to {kaggle_trimmed['date'].max().date()})")
print(f"Trendlyne rows: {len(trendlyne)} ({trendlyne['date'].min().date()} → {trendlyne['date'].max().date()})")

# Combine
combined = pd.concat([kaggle_trimmed, trendlyne], ignore_index=True)
combined = combined.sort_values('date').reset_index(drop=True)

# Final check
print(f"\nTotal rows: {len(combined)}")
print(f"Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")
print(f"\nRows per year:")
print(combined.groupby(combined['date'].dt.year).size())

combined.to_parquet("raw/fii_dii_flows.parquet", index=False)
print("\nD7 Complete — Saved ✅")