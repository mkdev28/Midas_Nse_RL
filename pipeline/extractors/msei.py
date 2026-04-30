import requests
import pandas as pd
from datetime import date, timedelta
import time
import os
import io

os.makedirs("raw", exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
})

def fetch_msei_fii_dii(date_obj):
    """MSEI publishes individual CSV per trading day"""
    d = date_obj.strftime("%d%m%Y")  # format: 01012022
    url = f"https://www.msei.in/downloads/equity-reports/fii-dii-activities/fii_dii_{d}.csv"
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200 and len(resp.content) > 100:
            df = pd.read_csv(io.StringIO(resp.text))
            df['date'] = date_obj
            return df
    except:
        pass
    return None

# Test with a few recent dates first
test_dates = [
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 3, 15),
    date(2023, 6, 1),
]

print("Testing MSEI endpoint...")
for d in test_dates:
    result = fetch_msei_fii_dii(d)
    if result is not None:
        print(f"✅ {d}: {len(result)} rows — columns: {result.columns.tolist()}")
    else:
        print(f"❌ {d}: No data")
    time.sleep(1)