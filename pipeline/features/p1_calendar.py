# pipeline/features/p1_calendar.py
import sys, os, pandas as pd
sys.path.append(r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse")
from config import RAW, PROC

os.makedirs(PROC, exist_ok=True)

nifty = pd.read_parquet(os.path.join(RAW, "nifty_index.parquet"))
nifty.index = pd.to_datetime(nifty.index)

trading_days = nifty.index.normalize().unique().sort_values()
trading_days = trading_days[(trading_days >= '2008-01-01') & (trading_days <= '2025-12-31')]

cal = pd.DataFrame({'date': trading_days})
cal['year']       = cal['date'].dt.year
cal['month']      = cal['date'].dt.month
cal['quarter']    = cal['date'].dt.quarter
cal['dayofweek']  = cal['date'].dt.dayofweek
cal['weekofyear'] = cal['date'].dt.isocalendar().week.astype(int)
cal['t_index']    = range(len(cal))

cal.to_parquet(os.path.join(PROC, "trading_calendar.parquet"), index=False)
print(f"P1 ✅ — {len(cal)} trading days saved")