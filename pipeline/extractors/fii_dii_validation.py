import pandas as pd

df = pd.read_parquet("raw/fii_dii_kaggle_clean.parquet")

print("=== SANITY CHECK VS KNOWN HISTORICAL EVENTS ===\n")

# Jan 2008 — FIIs sold ~17,326 crore total that month
jan_2008 = df[(df['date'].dt.year == 2008) & (df['date'].dt.month == 1)]
print(f"Jan 2008 FII net total: ₹{jan_2008['fii_net'].sum():.0f} crore")
print(f"  Expected: ~-17,326 crore (strongly negative) [ET report]\n")

# Oct 2008 GFC peak — FII net ~-14,248 crore
oct_2008 = df[(df['date'].dt.year == 2008) & (df['date'].dt.month == 10)]
print(f"Oct 2008 FII net total: ₹{oct_2008['fii_net'].sum():.0f} crore")
print(f"  Expected: ~-14,248 crore\n")

# 2020 COVID crash (Mar 2020) — FII should be strongly negative
mar_2020 = df[(df['date'].dt.year == 2020) & (df['date'].dt.month == 3)]
print(f"Mar 2020 FII net total: ₹{mar_2020['fii_net'].sum():.0f} crore")
print(f"  Expected: strongly negative (COVID panic selling)\n")

# 2021 — FII should be positive (post-COVID recovery rally)
y2021 = df[df['date'].dt.year == 2021]
print(f"Full 2021 FII net: ₹{y2021['fii_net'].sum():.0f} crore")
print(f"  Expected: positive (global liquidity rally)\n")

# DII counter-buying pattern — DII net should be opposite to FII during stress
print("DII net during FII sell months (should be positive = counter-buying):")
stress_months = df[df['fii_net'] < -500]
print(f"  When FII daily net < -500cr, DII avg net: ₹{stress_months['dii_net'].mean():.0f} crore")
print(f"  (Positive = DIIs buying when FIIs sell ✅)")