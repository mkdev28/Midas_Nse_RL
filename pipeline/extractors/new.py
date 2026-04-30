import requests
import pandas as pd
import time
import os

os.makedirs("raw", exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
})

# Warm up session — mandatory
print("Warming up NSE session...")
session.get("https://www.nseindia.com", timeout=15)
time.sleep(3)
session.get("https://www.nseindia.com/market-data/live-equity-market", timeout=15)
time.sleep(3)
print("Session ready ✅")

# Fetch in quarterly chunks
chunks = [
    ("01-01-2022", "31-03-2022"), ("01-04-2022", "30-06-2022"),
    ("01-07-2022", "30-09-2022"), ("01-10-2022", "31-12-2022"),
    ("01-01-2023", "31-03-2023"), ("01-04-2023", "30-06-2023"),
    ("01-07-2023", "30-09-2023"), ("01-10-2023", "31-12-2023"),
    ("01-01-2024", "31-03-2024"), ("01-04-2024", "30-06-2024"),
    ("01-07-2024", "30-09-2024"), ("01-10-2024", "31-12-2024"),
    ("01-01-2025", "31-03-2025"), ("01-04-2025", "30-06-2025"),
    ("01-07-2025", "30-09-2025"), ("01-10-2025", "31-12-2025"),
    ("01-01-2026", "24-04-2026"),
]

all_dfs = []
for start, end in chunks:
    url = f"https://www.nseindia.com/api/fiidiiTradeReact?startDate={start}&endDate={end}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data)
            all_dfs.append(df)
            print(f"✅ {start} → {end}: {len(df)} rows")
        else:
            print(f"❌ {start} → {end}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"❌ {start} → {end}: {e}")
    time.sleep(2)

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined)}")
    print(f"Columns: {combined.columns.tolist()}")
    print(combined.head(4))
    combined.to_csv("raw/fii_dii_nse_2022_2026.csv", index=False)
    print("Saved ✅")
else:
    print("\n⚠️ All requests failed — proceed to manual download below")