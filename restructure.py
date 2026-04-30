import shutil
import os

ROOT = r"c:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse"

# Create clean structure
dirs = [
    "data/raw", "data/processed", "data/external",
    "pipeline/extractors", "pipeline/features",
    "pipeline/validators", "pipeline/packaging",
    "checkpoints", "logs"
]
for d in dirs:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# ── Move parquets from midas_nse/raw/ → data/raw/ ──────────────────
src_raw = os.path.join(ROOT, "raw")
dst_raw = os.path.join(ROOT, "data", "raw")

parquets_to_move = [
    "fii_dii_flows.parquet",
    "fii_dii_kaggle_clean.parquet",
    "fii_dii_trendlyne_2022_2026.parquet",
    "gsec_10y_yield_raw.parquet",
]
for f in parquets_to_move:
    src = os.path.join(src_raw, f)
    dst = os.path.join(dst_raw, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved → data/raw/{f}")

# ── Move CSVs/TXT from midas_nse/raw/ → data/external/ ─────────────
dst_ext = os.path.join(ROOT, "data", "external")
externals = [
    "fii_dii_trendlyne.txt",
    "fii_dii_trendlyne_2022_2026.csv",
    "fii_dii_nse_2022_2026.csv",
    "rbi_repo_rate_raw.csv",
    "news_economic_times_rss.csv",
    "news_financial_classification.csv",
    "news_india_financial.csv",
    "news_moneycontrol_rss.csv",
    "news_nifty_huggingface.csv",
]
for f in externals:
    src = os.path.join(src_raw, f)
    dst = os.path.join(dst_ext, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved → data/external/{f}")

# ── Move loose files from midas_nse/ root → data/external/ ─────────
root_files = [
    "Fii Dii Trading activity.csv",
    "India 10-Year Bond Yield Historical Data.csv",
    "fii_dii_full_2008_2025.csv",
    "Major-Monetary-Policy-Rates-and-Reserve-Requirements-Bank-Rate-LAF-Repo-Reverse-Repo-SDF-and-MSF-Rates-CRR-SLR.xlsx",
]
for f in root_files:
    src = os.path.join(ROOT, f)
    dst = os.path.join(dst_ext, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved root → data/external/{f}")

# ── Delete empty midas_nse/raw/ if empty ────────────────────────────
if os.path.exists(src_raw) and not os.listdir(src_raw):
    os.rmdir(src_raw)
    print("Deleted empty midas_nse/raw/")
else:
    remaining = os.listdir(src_raw) if os.path.exists(src_raw) else []
    if remaining:
        print(f"midas_nse/raw/ still has: {remaining}")

print("\n✅ Migration complete")
print(f"\ndata/raw/ contents:")
for f in sorted(os.listdir(dst_raw)):
    size = os.path.getsize(os.path.join(dst_raw, f)) / 1024
    print(f"  {f} ({size:.1f} KB)")