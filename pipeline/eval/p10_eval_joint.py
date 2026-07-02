import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO, SAC

from pipeline.packaging.p9_gym_env import (
    MacroAllocatorEnv, SentimentModifierEnv, StockPickerEnv,
    MidasDataset, _MidasEncoder, coordinate
)
from pipeline.eval.p10_validate_a1_a3 import seed_everything

CKPT   = Path("checkpoints")
device = "cuda" if torch.cuda.is_available() else "cpu"
COST_PER_UNIT = (10 + 5) / 10_000   # 15 bps

def run_coordinator_pipeline(a1_model, a2_model, a3_model, encoder, seed=7):
    """
    Run the full A1 → A2 → A3 → Coordinator pipeline on the test split.
    Returns (sharpe, cagr, mdd, final_equity).
    """
    seed_everything(seed)
    test_ds = MidasDataset(split="test")

    a1_env = MacroAllocatorEnv(dataset=test_ds, encoder=encoder, device=device)
    a3_env = StockPickerEnv(dataset=test_ds, encoder=encoder, device=device)
    a2_env = SentimentModifierEnv(dataset=test_ds, a1_weights_fn=lambda t: a1_env.current_weights)

    obs_a1, _ = a1_env.reset(seed=seed)
    obs_a3, _ = a3_env.reset(seed=seed)
    obs_a2, _ = a2_env.reset(seed=seed)

    equity_curve  = [1.0]
    daily_returns = []
    equity        = 1.0
    prev_weights  = np.zeros(50, dtype=np.float32)
    done          = False

    while not done:
        action_a1, _ = a1_model.predict(obs_a1, deterministic=True)
        action_a3, _ = a3_model.predict(obs_a3, deterministic=True)
        action_a2, _ = a2_model.predict(obs_a2, deterministic=True)

        exp_a1 = np.exp(action_a1 - action_a1.max())
        w_a1   = (exp_a1 / exp_a1.sum()).astype(np.float32)

        exp_a3 = np.exp(action_a3 - action_a3.max())
        w_a3   = (exp_a3 / exp_a3.sum()).astype(np.float32)

        w_a2 = np.clip(action_a2, 0.5, 1.5).astype(np.float32)

        final_weights = coordinate(w_a1, w_a2, w_a3)

        t = a1_env.t
        stock_rets = test_ds.stock_returns[test_ds.stock_idx[t]].copy()

        gross_ret = float((final_weights * stock_rets).sum())
        turnover  = float(np.abs(final_weights - prev_weights).sum())
        net_ret   = float(np.clip(gross_ret - turnover * COST_PER_UNIT, -0.15, 0.15))
        if not np.isfinite(net_ret):
            net_ret = 0.0

        equity = equity * (1.0 + net_ret)
        daily_returns.append(net_ret)
        equity_curve.append(equity)
        prev_weights = final_weights

        # Advance all envs together
        a1_env.t += 1
        a2_env.t += 1
        a3_env.t += 1
        done = a1_env.t >= test_ds.T - 1

        if not done:
            obs_a1 = a1_env._get_obs()
            obs_a3 = a3_env._get_obs()
            obs_a2 = a2_env._get_obs()

    rets = pd.Series(daily_returns)
    eq   = pd.Series(equity_curve)

    std    = rets.std(ddof=1)
    sharpe = np.sqrt(252) * rets.mean() / std if std > 0 else 0.0
    n_yrs  = len(eq) / 252
    cagr   = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_yrs) - 1
    mdd    = (eq / eq.cummax() - 1.0).min()

    return sharpe, cagr, mdd, eq.iloc[-1]


def main():
    print("=" * 60)
    print("Joint Fine-Tuned Pipeline Evaluation (Unified)")
    print("Test Period: 2023-01-02 to 2025-12-29")
    print("Costs: 15 bps × L1 turnover per day")
    print("=" * 60)

    # ── Load Encoder ──────────────────────────────────────────────
    ckpt = torch.load(CKPT / "transformer_encoder.pt", map_location=device, weights_only=False)
    encoder = _MidasEncoder()
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device).eval()

    # ── Evaluate Joint Fine-Tuned Models ─────────────────────────
    print("\n[Loading Joint Fine-Tuned Models]")
    a1_joint = SAC.load(CKPT / "a1_unified_joint_final", device=device)
    a2_joint = PPO.load(CKPT / "a2_unified_joint_final", device="cpu")
    a3_joint = SAC.load(CKPT / "a3_unified_joint_final", device=device)

    seeds = [7, 11, 19]
    sharpes, cagrs, mdds, equities = [], [], [], []

    print("\nRunning 3-seed evaluation...")
    for seed in seeds:
        s, c, m, e = run_coordinator_pipeline(a1_joint, a2_joint, a3_joint, encoder, seed)
        sharpes.append(s)
        cagrs.append(c)
        mdds.append(m)
        equities.append(e)
        print(f"  Seed {seed:2d} -> Sharpe: {s:.3f} | CAGR: {c*100:.1f}% | MDD: {m*100:.1f}% | Final Equity: {e:.3f}")

    print("-" * 60)
    print(f"  MEAN   -> Sharpe: {np.mean(sharpes):.3f} | CAGR: {np.mean(cagrs)*100:.1f}% | MDD: {np.mean(mdds)*100:.1f}% | Final Equity: {np.mean(equities):.3f}")

    # ── Summary vs Baselines ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON TABLE (all test set, 15 bps costs)")
    print("=" * 60)
    print(f"{'Model':<40} {'Sharpe':>7} {'CAGR':>8} {'MDD':>9}")
    print("=" * 67)
    print("-- RL Agents (Unified Architecture) -------------------------")
    print(f"{'A1 Macro Unified':<40} {'1.540':>7} {'10.3%':>8} {'-4.7%':>9}")
    print(f"{'A3 Stock Picker Unified (stock-only)':<40} {'1.863':>7} {'23.7%':>8} {'-14.5%':>9}")
    print(f"{'A1+A2 Macro+Sentiment Unified':<40} {'1.647':>7} {'8.7%':>8} {'-3.5%':>9}")
    print(f"{'Joint A1+A2+A3 Unified (100k steps)':<40} {'1.635':>7} {'20.4%':>8} {'-18.2%':>9}")
    print(f"{'Joint A1+A2+A3 Unified (200k steps)':<40} {np.mean(sharpes):>7.3f} {np.mean(cagrs)*100:>7.1f}% {np.mean(mdds)*100:>8.1f}%")
    print("-- RL Agents (C2 Cross-Modal Architecture) ------------------")
    print(f"{'A1 Macro C2':<40} {'0.661':>7} {'4.8%':>8} {'-12.0%':>9}")
    print(f"{'A3 Stock Picker C2 (stock-only)':<40} {'1.522':>7} {'19.0%':>8} {'-17.2%':>9}")
    print(f"{'A1+A2 Macro+Sentiment C2':<40} {'< 0.0':>7} {'N/A':>8} {'N/A':>9}")
    print(f"{'Joint A1+A2+A3 C2 (100k steps)':<40} {'1.431':>7} {'17.8%':>8} {'-19.2%':>9}")
    print("-- Classical Baselines (stock-only, comparable to A3) -------")
    print(f"{'Equal Weight (50 stocks)':<40} {'1.876':>7} {'23.6%':>8} {'-15.8%':>9}")
    print(f"{'Markowitz MVO (50 stocks)':<40} {'1.881':>7} {'~24%':>8} {'~-14%':>9}")
    print(f"{'Momentum (monthly rebal)':<40} {'1.871':>7} {'~22%':>8} {'~-14%':>9}")
    print(f"{'Vol Weighted (weekly rebal)':<40} {'1.950':>7} {'~26%':>8} {'~-13%':>9}")
    print("=" * 67)
    print()
    print("NOTE: A3 vs classical baselines is the correct apples-to-apples")
    print("      comparison (all stock-only). Joint pipeline is macro-level")
    print("      so its CAGR/MDD reflect macro + stock blend together.")


if __name__ == "__main__":
    main()
