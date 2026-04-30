import sys, os
import pandas as pd
sys.path.append(r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse")
from config import RAW, PROC, EXT

cal = pd.read_parquet(os.path.join(PROC, "trading_calendar.parquet"))
cal['date'] = pd.to_datetime(cal['date'])

def flatten_yfinance(df, prefix):
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df.columns = [f"{prefix}_{col[0].lower()}" for col in df.columns]
    df = df.reset_index().rename(columns={'index': 'date', 'Date': 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    return df

def align(df, value_cols, source_name):
    df = df.drop_duplicates(subset='date')
    merged = cal[['date']].merge(df[['date'] + value_cols], on='date', how='left')
    merged[value_cols] = merged[value_cols].ffill()
    nulls = merged[value_cols].isnull().sum().sum()
    print(f"  {source_name}: {len(df)} src rows → {len(merged)} aligned | nulls: {nulls}")
    return merged

print("Aligning all sources...\n")

# D2 — NIFTY Index
nifty_aligned = align(flatten_yfinance(pd.read_parquet(os.path.join(RAW, "nifty_index.parquet")), "nifty"),
                      ["nifty_close","nifty_high","nifty_low","nifty_open","nifty_volume"], "NIFTY Index")

# D3 — VIX
vix_aligned = align(flatten_yfinance(pd.read_parquet(os.path.join(RAW, "india_vix.parquet")), "vix"),
                    ["vix_close"], "India VIX")

# D4 — Gold
gold_aligned = align(flatten_yfinance(pd.read_parquet(os.path.join(RAW, "gold.parquet")), "gold"),
                     ["gold_close"], "Gold")

# D5 — Crude
crude_aligned = align(flatten_yfinance(pd.read_parquet(os.path.join(RAW, "crude.parquet")), "crude"),
                      ["crude_close"], "Crude Oil")

# D6 — INR/USD
inrusd_aligned = align(flatten_yfinance(pd.read_parquet(os.path.join(RAW, "inr_usd.parquet")), "inrusd"),
                       ["inrusd_close"], "INR/USD")

# D7 — FII/DII
fii = pd.read_parquet(os.path.join(RAW, "fii_dii_flows.parquet"))
fii['date'] = pd.to_datetime(fii['date']).dt.normalize()
fii_aligned = align(fii, ["fii_net","dii_net","fii_buy_value","fii_sell_value",
                           "dii_buy_value","dii_sell_value","fii_dii_available"], "FII/DII")

# D8 — Repo Rate
repo = pd.read_csv(os.path.join(EXT, "rbi_repo_rate_raw.csv"))
repo['date'] = pd.to_datetime(repo['date']).dt.normalize()  # no dayfirst needed
repo_aligned = align(repo, ["repo_rate"], "Repo Rate")


# D9 — G-Sec (in EXT)
gsec = pd.read_csv(os.path.join(EXT, "India 10-Year Bond Yield Historical Data.csv"))
gsec = gsec.rename(columns={"Date": "date", "Price": "gsec_10y_yield"})
gsec['date'] = pd.to_datetime(gsec['date'], dayfirst=True).dt.normalize()
gsec['gsec_10y_yield'] = pd.to_numeric(gsec['gsec_10y_yield'].astype(str).str.replace(',',''), errors='coerce')
gsec_aligned = align(gsec, ["gsec_10y_yield"], "G-Sec 10Y")

# ── Merge all onto calendar ──────────────────────────────────────────
print("\nMerging...")
master = cal.copy()
for df in [nifty_aligned, vix_aligned, gold_aligned, crude_aligned,
           inrusd_aligned, fii_aligned, repo_aligned, gsec_aligned]:
    cols = [c for c in df.columns if c != 'date']
    master = master.merge(df[['date'] + cols], on='date', how='left')

# Fill remaining nulls
master['vix_close'] = master['vix_close'].bfill()
master = master.ffill().bfill()

print(f"\nMaster shape: {master.shape}")
print(f"Total nulls: {master.isnull().sum().sum()}")
print(f"Columns: {master.columns.tolist()}")

master.to_parquet(os.path.join(PROC, "master_aligned.parquet"), index=False)
print("\nP2 ✅ — master_aligned.parquet saved")