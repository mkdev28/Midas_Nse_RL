import pandas as pd
import os

os.makedirs("raw", exist_ok=True)

df = pd.read_csv(r"C:\Users\mohit\Downloads\India 10-Year Bond Yield Historical Data.csv")
print(df.head(3))
print(df.dtypes)

df = df.rename(columns={"Date": "date", "Price": "gsec_10y_yield"})
df['date'] = pd.to_datetime(df['date'],dayfirst=True)
df['gsec_10y_yield'] = pd.to_numeric(
    df['gsec_10y_yield'].astype(str).str.replace(',', ''), errors='coerce'
)
df = df[['date', 'gsec_10y_yield']].dropna()
df = df.sort_values('date').reset_index(drop=True)

print(f"Rows: {len(df)}")
print(f"Range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Yield range: {df['gsec_10y_yield'].min()}% → {df['gsec_10y_yield'].max()}%")

df.to_parquet("raw/gsec_10y_yield_raw.parquet")
print("Saved ✅")