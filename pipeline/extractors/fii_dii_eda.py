import pandas as pd

# Adjust filename to yours
df = pd.read_csv(r"C:\Users\mohit\Downloads\Fii Dii Trading activity.csv")


import os

os.makedirs("raw", exist_ok=True)



# Fix: parse with dayfirst=True and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

# Rename columns to standard schema
df = df.rename(columns={
    'Date': 'date',
    'FII_Gross_Purchase': 'fii_buy_value',
    'FII_Gross_Sales': 'fii_sell_value',
    'FII_Net_Purchase/Sales': 'fii_net',
    'DII_Gross_Purchase': 'dii_buy_value',
    'DII_Gross_Sales': 'dii_sell_value',
    'DII_Net_Purchase/Sales': 'dii_net'
})

# Drop the 10 Jan-1 holiday rows (not real trading days)
df = df[~((df['date'].dt.month == 1) & (df['date'].dt.day == 1))]

# Verify
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"\nRows per year:")
print(df.groupby(df['date'].dt.year).size())
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nSanity check — FII net in 2008 (should be strongly negative):")
print(df[df['date'].dt.year == 2008]['fii_net'].describe())

df.to_parquet("raw/fii_dii_kaggle_clean.parquet", index=False)
print("\nSaved ✅")