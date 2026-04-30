import pandas as pd
import re
import os

os.makedirs("raw", exist_ok=True)

with open("raw/fii_dii_trendlyne.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Clean up tabs and split into lines
lines = [l.strip() for l in content.split('\n')]

rows = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Date line pattern: "3 Jan 2022", "24 Feb 2023" etc.
    if re.match(r'^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$', line):
        # Collect next 6 non-empty numeric lines
        vals = []
        j = i + 1
        while j < len(lines) and len(vals) < 6:
            v = lines[j].replace(',', '').strip()
            if v:  # skip blank lines
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            j += 1
        
        if len(vals) == 6:
            rows.append({
                'date': line,
                'fii_buy_value': vals[0],
                'fii_sell_value': vals[1],
                'fii_net': vals[2],
                'dii_net': vals[3],
                'dii_sell_value': vals[4],
                'dii_buy_value': vals[5],
            })
            i = j
        else:
            i += 1  # incomplete row, skip
    else:
        i += 1

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], format='%d %b %Y', errors='coerce')
df = df.dropna(subset=['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"\nRows per year:")
print(df.groupby(df['date'].dt.year).size())
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
print(f"\nSanity — Jan 2022 FII net (expect strongly negative, Fed hike fears):")
print(f"  ₹{df[df['date'].dt.strftime('%Y-%m') == '2022-01']['fii_net'].sum():.1f} crore")

df.to_csv("raw/fii_dii_trendlyne_2022_2026.csv", index=False)
df.to_parquet("raw/fii_dii_trendlyne_2022_2026.parquet", index=False)
print("\nSaved ✅")