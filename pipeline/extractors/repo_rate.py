import pandas as pd
import os

os.makedirs("raw", exist_ok=True)

df_raw = pd.read_excel(
    r"C:\Users\mohit\Downloads\Major Monetary Policy Rates and Reserve Requirements - Bank Rate, LAF (Repo, Reverse Repo, SDF and MSF) Rates, CRR & SLR.xlsx",      # Column B = date, Column C = Repo Rate
    header=None
)

# Data starts at row 8, column 1 = date, column 2 = repo rate
# Filter rows where column 1 is a real datetime object
import datetime
mask = df_raw[1].apply(lambda x: isinstance(x, datetime.datetime))
data = df_raw[mask][[1, 2]].copy()
data.columns = ["date", "repo_rate"]

# Drop '-' (no change rows) and NaN
data = data[data["repo_rate"] != "-"]
data = data[data["repo_rate"].notna()]
data["repo_rate"] = pd.to_numeric(data["repo_rate"], errors="coerce")
data = data.dropna(subset=["repo_rate"])

# Filter 2008 onwards, sort ascending
data = data[data["date"] >= datetime.datetime(2008, 1, 1)]
data = data.sort_values("date").reset_index(drop=True)

# Validate
print(f"Rows: {len(data)}")
print(f"Date range: {data['date'].min().date()} → {data['date'].max().date()}")
print(f"Repo rate range: {data['repo_rate'].min()}% → {data['repo_rate'].max()}%")
print(data.to_string())

data.to_csv("raw/rbi_repo_rate_raw.csv", index=False)
print("\nSaved to raw/rbi_repo_rate_raw.csv ✅")