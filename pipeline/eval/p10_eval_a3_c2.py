"""
p10_eval_a3_c2.py — Evaluate A3 C2 checkpoints on the test set

Sweeps all 3 C2 checkpoints (50K, 100K, 150K) plus the unified A3 baseline (50K)
and prints a comparison table.  Use the results to pick the production checkpoint.

Obs : 856 = Z(256) + 50 × 12 stock features
Act : 50 stock weights
Env : StockPickerEnv (test split, 2023-01-02 → 2025-12-29, 738 days)
"""

import sys, os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from models.cross_modal_transformer import CrossModalTransformer
from pipeline.packaging.p9_gym_env import StockPickerEnv, MidasDataset, _MidasEncoder

# Transaction costs — must match p10_validate_a1_a3.py exactly for paper comparability
# 10 bps transaction + 5 bps slippage applied per unit of L1 turnover each day
TXN_COST_BPS  = 10
SLIPPAGE_BPS  = 5
COST_PER_UNIT = (TXN_COST_BPS + SLIPPAGE_BPS) / 10_000   # = 0.0015

# ── Helpers ───────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_episode(env, model, seed: int) -> dict:
    """Roll out one deterministic episode and return metrics dict.

    Transaction costs: 15 bps (10 txn + 5 slippage) × L1 turnover per step.
    This matches p10_validate_a1_a3.py exactly so all numbers are comparable.
    """
    seed_everything(seed)
    obs, _ = env.reset(seed=seed)
    equity        = 1.0
    equity_curve  = [1.0]
    daily_returns = []
    prev_weights  = np.full(50, 1/50, dtype=np.float64)   # uniform at start (matches env init)
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        out = env.step(action)
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated

        gross_ret    = float(info.get("port_ret", reward))
        curr_weights = np.asarray(info.get("weights", prev_weights), dtype=np.float64)

        # Apply 15 bps × L1 turnover — same formula as p10_validate_a1_a3.py
        turnover = float(np.abs(curr_weights - prev_weights).sum())
        cost     = turnover * COST_PER_UNIT
        net_ret  = float(np.clip(gross_ret - cost, -0.15, 0.15))
        if not np.isfinite(net_ret):
            net_ret = 0.0

        equity *= (1.0 + net_ret)
        daily_returns.append(net_ret)
        equity_curve.append(equity)
        prev_weights = curr_weights

    rets = np.array(daily_returns, dtype=np.float64)
    eq   = np.array(equity_curve,  dtype=np.float64)

    # Core metrics
    excess = rets          # Rf ≈ 0 for daily
    std    = excess.std(ddof=1)
    sharpe = np.sqrt(252) * excess.mean() / std if std > 0 else 0.0

    n_years = len(rets) / 252
    cagr    = (eq[-1] / eq[0]) ** (1 / n_years) - 1

    running_max = np.maximum.accumulate(eq)
    mdd         = float((eq / running_max - 1.0).min())

    # Sortino (downside std)
    down = excess[excess < 0]
    sortino = np.sqrt(252) * excess.mean() / down.std(ddof=1) if len(down) > 1 else 0.0

    # Calmar
    calmar = (cagr / abs(mdd)) if abs(mdd) > 1e-8 else 0.0

    return dict(sharpe=sharpe, sortino=sortino, cagr=cagr, mdd=mdd,
                calmar=calmar, final_eq=float(eq[-1]))


def eval_checkpoint(ckpt_path: Path, env, is_c2: bool,
                    c2_encoder=None, unified_encoder=None,
                    device: str = "cpu", seeds=(7, 11, 19)) -> dict:
    """Load a checkpoint and run multi-seed evaluation."""
    if not ckpt_path.exists():
        return None

    # Env uses whichever encoder was passed at construction time.
    # We rebuild the env so the right encoder is embedded.
    ds = env.ds          # reuse same dataset object
    if is_c2:
        test_env = StockPickerEnv(ds, c2_encoder, device)
    else:
        test_env = StockPickerEnv(ds, unified_encoder, device)

    model = SAC.load(str(ckpt_path), device=device)

    all_metrics = [run_episode(test_env, model, s) for s in seeds]

    return {k: float(np.mean([m[k] for m in all_metrics]))
            for k in all_metrics[0]}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("A3 C2 Architecture -- Checkpoint Sweep Evaluation")
    print("Test period: 2023-01-02 to 2025-12-29  |  738 trading days")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT   = PROJECT_ROOT / "checkpoints"
    SEEDS  = [7, 11, 19]

    # ── Load C2 Encoder ───────────────────────────────────────────────────────
    c2_ckpt_path = CKPT / "c2" / "transformer_c2_encoder.pt"
    c2_ckpt      = torch.load(c2_ckpt_path, map_location=device, weights_only=False)
    config       = c2_ckpt["config"]

    c2_encoder = CrossModalTransformer(
        d_enc    = config["d_enc"],
        d_out    = config["d_out"],
        nhead    = config["nhead"],
        n_layers = config["n_layers"],
        window   = config["window"],
    ).to(device)
    raw     = c2_ckpt["encoder_state"]
    stripped = {k.replace("encoder.", "", 1): v for k, v in raw.items()}
    c2_encoder.load_state_dict(stripped)
    c2_encoder.eval()
    for p in c2_encoder.parameters():
        p.requires_grad = False
    print(f"C2 encoder loaded ({len(c2_ckpt['feature_cols'])} features) [OK]")

    # ── Load Unified Encoder (for baseline run) ───────────────────────────────
    u_ckpt      = torch.load(CKPT / "transformer_encoder.pt",
                             map_location=device, weights_only=False)
    u_encoder   = _MidasEncoder().to(device)
    u_encoder.load_state_dict(u_ckpt["encoder_state"])
    u_encoder.eval()
    for p in u_encoder.parameters():
        p.requires_grad = False
    print("Unified encoder loaded [OK]")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading test dataset...")
    test_ds = MidasDataset(split="test")

    # Placeholder env (encoder will be swapped per run inside eval_checkpoint)
    dummy_env = StockPickerEnv(test_ds, c2_encoder, device)

    # ── Checkpoint Map ────────────────────────────────────────────────────────
    checkpoints = [
        # (label,                         path,                                        is_c2)
        ("A3 C2   50K",  CKPT / "c2" / "a3" / "a3_c2_sac_50000_steps.zip",   True),
        ("A3 C2  100K",  CKPT / "c2" / "a3" / "a3_c2_sac_100000_steps.zip",  True),
        ("A3 C2  150K",  CKPT / "c2" / "a3" / "a3_c2_sac_150000_steps.zip",  True),
        ("A3 C2  Final", CKPT / "c2" / "a3" / "a3_c2_sac_final.zip",         True),
        ("A3 Unified 50K (baseline)", CKPT / "a3_sac_50000_steps.zip",        False),
    ]

    # ── Run Evaluation ────────────────────────────────────────────────────────
    print("\nRunning evaluation (3 seeds per checkpoint)...")
    rows = []
    for label, path, is_c2 in checkpoints:
        if not path.exists():
            print(f"  SKIP (not found): {path.name}")
            continue
        print(f"  Evaluating: {label} ...", end=" ", flush=True)
        metrics = eval_checkpoint(
            path, dummy_env, is_c2,
            c2_encoder=c2_encoder,
            unified_encoder=u_encoder,
            device=device,
            seeds=SEEDS,
        )
        if metrics:
            rows.append({"Checkpoint": label, **metrics})
            print(f"Sharpe {metrics['sharpe']:.3f} | CAGR {metrics['cagr']:.1%} | MDD {metrics['mdd']:.1%}")
        else:
            print("FAILED")

    # ── Results Table ─────────────────────────────────────────────────────────
    if not rows:
        print("\nNo checkpoints evaluated — check paths.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("RESULTS TABLE (sorted by Sharpe, mean of 3 seeds)")
    print("=" * 65)
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
        formatters={"cagr": lambda x: f"{x:.1%}", "mdd": lambda x: f"{x:.1%}"},
    ))

    best = df.iloc[0]
    print("\n" + "-" * 65)
    print(f"BEST CHECKPOINT : {best['Checkpoint']}")
    print(f"  Sharpe : {best['sharpe']:.3f}")
    print(f"  Sortino: {best['sortino']:.3f}")
    print(f"  CAGR   : {best['cagr']:.1%}")
    print(f"  MDD    : {best['mdd']:.1%}")
    print(f"  Calmar : {best['calmar']:.3f}")
    print(f"  Final Equity: {best['final_eq']:.3f}")

    print("\n[Baseline Targets -- all after 15 bps x turnover costs, same methodology]")
    print("A3 Unified 50K  : Sharpe 1.862 | CAGR 23.1% | MDD -14.5%  (from p10_validate_a1_a3.py)")
    print("A1 C2 Production: Sharpe 1.981 | CAGR 26.0% | MDD  -8.1%  (from p10_eval_a1_c2.py — NO costs applied, A1 has low turnover ~0.16)")
    print("Equal Weight    : Sharpe 1.876 | CAGR 23.6% | MDD -15.8%  (from p11_baselines_classical.py)")
    print("\nNOTE: This script now applies 15 bps x turnover costs, same as p10_validate_a1_a3.py.")
    print("      The A3 Unified 50K result above (1.862) is the ground truth to beat for A3.")

    # ── Save Results ──────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "results" / "p10_eval_a3_c2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "a3_c2_checkpoint_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")

if __name__ == "__main__":
    main()
