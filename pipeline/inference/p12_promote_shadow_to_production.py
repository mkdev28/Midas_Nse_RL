"""
MIDAS-NSE — Shadow -> Production Promotion Script
=================================================
MANUAL ONLY. Run this only when you want to consider promoting
the shadow A3 model to production.

This script:
  1. Runs a full offline backtest of BOTH shadow and production A3
     on the last 6+ months of live data (from rolling_returns.csv)
  2. Compares: Sharpe, Max Drawdown, Calmar, Turnover
  3. Prints a clear comparison table
  4. Asks for EXPLICIT user confirmation ("yes" typed) before writing

NEVER run this automatically. It is a deliberate, research-grade decision.

Usage:
  python pipeline/inference/p12_promote_shadow_to_production.py
"""

import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT_DIR       = Path("checkpoints")
LIVE_DIR       = CKPT_DIR / "live"
SHADOW_CKPT    = LIVE_DIR / "shadow_model" / "a3_shadow"
PROD_CKPT      = CKPT_DIR / "a3_unified_joint_final"     # PRODUCTION — read only here
PROD_COPY_CKPT = LIVE_DIR / "prod_baseline" / "a3_prod_copy"
ROLLING_LOG    = LIVE_DIR / "rolling_returns.csv"
SHADOW_LOG     = LIVE_DIR / "shadow_log.csv"
PROMOTION_LOG  = LIVE_DIR / "promotion_history.csv"

MIN_BACKTEST_DAYS = 126   # ~6 months of trading days required before promotion
SHARPE_THRESHOLD  = 0.05  # Shadow must beat production by at least this to recommend promotion


def compute_metrics(returns: pd.Series, label: str) -> dict:
    """Compute Sharpe, MDD, Calmar, and avg turnover from return series."""
    r = returns.dropna()
    if len(r) < 30:
        print(f"  [{label}] Insufficient data ({len(r)} days).")
        return {}

    std    = r.std()
    sharpe = float(r.mean() / std * np.sqrt(252)) if std > 1e-8 else 0.0

    equity = (1 + r).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    mdd    = float(dd.min())

    cagr   = float((equity.iloc[-1] ** (252 / len(r))) - 1)
    calmar = abs(cagr / mdd) if mdd < -1e-8 else 0.0

    return {
        "label":   label,
        "days":    len(r),
        "sharpe":  round(sharpe, 4),
        "mdd":     round(mdd,    4),
        "cagr":    round(cagr,   4),
        "calmar":  round(calmar, 4),
    }


def run_promotion_check():
    print("=" * 65)
    print("  MIDAS-NSE — Shadow A3 -> Production Promotion Check")
    print("=" * 65)
    print()

    # ── Load rolling log ───────────────────────────────────────────────────
    if not ROLLING_LOG.exists():
        print("ERROR: rolling_returns.csv not found.")
        print("Run continuous_learner.py for at least one day first.")
        return

    df = pd.read_csv(ROLLING_LOG, parse_dates=["date"])

    if len(df) < MIN_BACKTEST_DAYS:
        print(f"INSUFFICIENT HISTORY: {len(df)} days < {MIN_BACKTEST_DAYS} required.")
        print(f"Run continuous_learner.py for at least {MIN_BACKTEST_DAYS} trading days.")
        print("Promotion blocked.")
        return

    print(f"Backtest period: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Days available:  {len(df)}")
    print()

    # ── Compute metrics ────────────────────────────────────────────────────
    shadow_metrics = compute_metrics(df["shadow_ret"], "Shadow A3")
    prod_metrics   = compute_metrics(df["prod_ret"],   "Production A3")

    if not shadow_metrics or not prod_metrics:
        print("ERROR: Could not compute metrics. Promotion blocked.")
        return

    # ── Print comparison table ─────────────────────────────────────────────
    print("  COMPARISON TABLE (Full Backtest Period)")
    print("-" * 65)
    print(f"  {'Metric':<20} {'Shadow A3':>15} {'Production A3':>15} {'Delta':>10}")
    print("-" * 65)
    metrics_to_show = [
        ("Sharpe Ratio",   "sharpe", "+.4f"),
        ("Max Drawdown",   "mdd",    "+.4f"),
        ("CAGR",           "cagr",   "+.4f"),
        ("Calmar Ratio",   "calmar", "+.4f"),
        ("Days evaluated", "days",   "d"),
    ]
    for label, key, fmt in metrics_to_show:
        sv = shadow_metrics.get(key, "N/A")
        pv = prod_metrics.get(key, "N/A")
        if isinstance(sv, float) and isinstance(pv, float):
            delta = sv - pv
            delta_str = f"{delta:{fmt}}" if fmt != "d" else f"{delta:+d}"
        else:
            delta_str = "N/A"
        sv_str = f"{sv:{fmt}}" if isinstance(sv, (int, float)) else str(sv)
        pv_str = f"{pv:{fmt}}" if isinstance(pv, (int, float)) else str(pv)
        print(f"  {label:<20} {sv_str:>15} {pv_str:>15} {delta_str:>10}")
    print("-" * 65)

    # ── Recommendation ─────────────────────────────────────────────────────
    sharpe_delta = shadow_metrics["sharpe"] - prod_metrics["sharpe"]
    mdd_improved = shadow_metrics["mdd"]    > prod_metrics["mdd"]   # less negative = better

    print()
    if sharpe_delta >= SHARPE_THRESHOLD and mdd_improved:
        recommendation = "RECOMMEND PROMOTION"
        reason = (f"Shadow Sharpe is {sharpe_delta:+.4f} above production "
                  f"and MDD is improved ({shadow_metrics['mdd']:.4f} vs {prod_metrics['mdd']:.4f}).")
    elif sharpe_delta >= SHARPE_THRESHOLD:
        recommendation = "CONDITIONALLY RECOMMEND (check MDD)"
        reason = (f"Shadow Sharpe is {sharpe_delta:+.4f} above production, "
                  f"but MDD worsened ({shadow_metrics['mdd']:.4f} vs {prod_metrics['mdd']:.4f}).")
    elif sharpe_delta >= 0:
        recommendation = "MARGINAL — YOUR CALL"
        reason = (f"Shadow Sharpe is {sharpe_delta:+.4f} above production "
                  f"(below {SHARPE_THRESHOLD} threshold). Risk: regression after promotion.")
    else:
        recommendation = "DO NOT PROMOTE"
        reason = (f"Shadow Sharpe is {sharpe_delta:+.4f} BELOW production. "
                  f"Shadow is underperforming. Continue learning.")

    print(f"  RECOMMENDATION: {recommendation}")
    print(f"  REASON:         {reason}")
    print()

    # ── Shadow log summary ─────────────────────────────────────────────────
    if SHADOW_LOG.exists():
        df_log = pd.read_csv(SHADOW_LOG)
        tail_risk_days = df_log.get("is_tail_risk_day", pd.Series([])).sum()
        gate_passed    = df_log["action_taken"].str.contains("PASSED").sum()
        gate_failed    = df_log["action_taken"].str.contains("FAILED").sum()
        print(f"  Shadow log summary ({len(df_log)} days):")
        print(f"    Tail-risk days:   {int(tail_risk_days)}")
        print(f"    Gate passed:      {int(gate_passed)}")
        print(f"    Gate failed:      {int(gate_failed)}")
        print()

    # ── Explicit user confirmation ─────────────────────────────────────────
    if "DO NOT PROMOTE" in recommendation:
        print("Promotion blocked by recommendation. Exiting.")
        return

    print("=" * 65)
    print("  MANUAL PROMOTION REQUIRED")
    print("=" * 65)
    print(f"  Shadow checkpoint: {SHADOW_CKPT}.zip")
    print(f"  Production ckpt:   {PROD_CKPT}.zip (will be overwritten)")
    print()
    print("  ⚠️  This is IRREVERSIBLE from this script.")
    print("  ⚠️  The original production checkpoint is immutable at:")
    print(f"       {PROD_COPY_CKPT}.zip")
    print()
    confirm = input("  Type 'PROMOTE' (all caps) to confirm promotion, or anything else to cancel: ")
    print()

    if confirm.strip() != "PROMOTE":
        print("  Promotion CANCELLED. No files changed.")
        return

    # ── Perform promotion ──────────────────────────────────────────────────
    shadow_zip = Path(str(SHADOW_CKPT) + ".zip")
    prod_zip   = Path(str(PROD_CKPT)   + ".zip")

    # Back up current production before overwriting
    backup_path = Path(str(PROD_CKPT)
                       + f"_backup_{datetime.today().strftime('%Y%m%d')}.zip")
    shutil.copy2(str(prod_zip), str(backup_path))
    print(f"  Production backed up to: {backup_path}")

    # Copy shadow → production
    shutil.copy2(str(shadow_zip), str(prod_zip))
    print(f"  ✅ Shadow promoted to production: {prod_zip}")

    # Log the promotion
    promo_record = {
        "date":          datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "shadow_sharpe": shadow_metrics["sharpe"],
        "prod_sharpe":   prod_metrics["sharpe"],
        "sharpe_delta":  round(sharpe_delta, 4),
        "shadow_mdd":    shadow_metrics["mdd"],
        "prod_mdd":      prod_metrics["mdd"],
        "shadow_cagr":   shadow_metrics["cagr"],
        "days_evaluated": shadow_metrics["days"],
        "backup_path":   str(backup_path),
    }
    pd.DataFrame([promo_record]).to_csv(
        PROMOTION_LOG, mode="a",
        header=not PROMOTION_LOG.exists(), index=False
    )
    print(f"  Promotion logged to: {PROMOTION_LOG}")
    print()
    print("  IMPORTANT: Re-run the P10/P11 evaluation suite to verify")
    print("  the promoted model on the full test set before publishing.")


if __name__ == "__main__":
    run_promotion_check()
