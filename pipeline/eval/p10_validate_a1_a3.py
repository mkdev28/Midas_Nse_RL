from __future__ import annotations

import os
import sys
import json
import math
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import SAC

from pipeline.packaging.p9_gym_env import (
    MacroAllocatorEnv,
    StockPickerEnv,
    MidasDataset,
    load_encoder,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
OUT_DIR = PROJECT_ROOT / "results" / "p10_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Checkpoint sweep lists ────────────────────────────────────────────
# Each entry: (label_for_output, path, agent_family, timestep_or_None)
# agent_family must be either "A1" or "A3" — used to select the right env.
A1_CHECKPOINTS: List[Tuple[str, Path, str, Any]] = [
    ("A1_unified_best",    CKPT_DIR / "a1_unified_fixed_best" / "best_model.zip", "A1", None),
    ("A1_unified_50k",   CKPT_DIR / "a1_unified_fixed_50000_steps.zip",  "A1", 50_000),
    ("A1_unified_100k",  CKPT_DIR / "a1_unified_fixed_100000_steps.zip", "A1", 100_000),
    ("A1_unified_150k",  CKPT_DIR / "a1_unified_fixed_150000_steps.zip", "A1", 150_000),
    ("A1_unified_200k",  CKPT_DIR / "a1_unified_fixed_200000_steps.zip", "A1", 200_000),
    ("A1_unified_250k",  CKPT_DIR / "a1_unified_fixed_250000_steps.zip", "A1", 250_000),
    ("A1_unified_300k",  CKPT_DIR / "a1_unified_fixed_300000_steps.zip", "A1", 300_000),
]

A3_CHECKPOINTS: List[Tuple[str, Path, str, Any]] = [
    ("A3_best",    CKPT_DIR / "a3_best" / "best_model.zip", "A3", None),
    ("A3_50000",   CKPT_DIR / "a3_sac_50000_steps.zip",  "A3", 50_000),
    ("A3_100000",  CKPT_DIR / "a3_sac_100000_steps.zip", "A3", 100_000),
    ("A3_150000",  CKPT_DIR / "a3_sac_150000_steps.zip", "A3", 150_000),
    ("A3_200000",  CKPT_DIR / "a3_sac_200000_steps.zip", "A3", 200_000),
    ("A3_250000",  CKPT_DIR / "a3_sac_250000_steps.zip", "A3", 250_000),
    ("A3_300000",  CKPT_DIR / "a3_sac_300000_steps.zip", "A3", 300_000),
    ("A3_350000",  CKPT_DIR / "a3_sac_350000_steps.zip", "A3", 350_000),
    ("A3_400000",  CKPT_DIR / "a3_sac_400000_steps.zip", "A3", 400_000),
    ("A3_450000",  CKPT_DIR / "a3_sac_450000_steps.zip", "A3", 450_000),
    ("A3_500000",  CKPT_DIR / "a3_sac_500000_steps.zip", "A3", 500_000),
    ("A3_final",   CKPT_DIR / "a3_sac_final.zip",         "A3", None),
]

ALL_CHECKPOINTS = A1_CHECKPOINTS + A3_CHECKPOINTS
ENCODER_CKPT = CKPT_DIR / "transformer_encoder.pt"

TEST_PARQUET = DATA_DIR / "test.parquet"
X_TEST_TECH = DATA_DIR / "X_test_technical.npy"
STOCK_META = DATA_DIR / "stock_features_meta.pkl"

SEEDS = [7, 11, 19]
RISK_FREE_RATE_ANNUAL = 0.0
TRADING_DAYS = 252

# Keep these explicit and documented.
TXN_COST_BPS = 10      # 10 bps = 0.10%
SLIPPAGE_BPS = 5       # 5 bps = 0.05%

# Update only if your env import/class path differs.
ENV_IMPORT = "pipeline.packaging.p9_gym_env"
A1_ENV_CLASS = "MacroAllocatorEnv"
A3_ENV_CLASS = "StockPickerEnv"

# After the p9_gym_env.py fix (un-normalizing col 9 in MidasDataset),
# info['port_ret'] is a true fractional daily return for both A1 and A3.
# No manual rescaling is needed here.

# ---------------------------------------------------------------------
# SEEDING
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def try_seed_env(env: Any, seed: int) -> Tuple[Any, Dict[str, Any]]:
    try:
        out = env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}
    except TypeError:
        pass

    try:
        if hasattr(env, "seed"):
            env.seed(seed)
    except Exception:
        pass

    try:
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
    except Exception:
        pass

    try:
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    except Exception:
        pass

    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def import_env_class(class_name: str):
    import importlib
    module = importlib.import_module(ENV_IMPORT)
    return getattr(module, class_name)


def safe_step(env: Any, action: np.ndarray):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    raise ValueError("Unexpected env.step output format")


def annualized_sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS
    excess = returns - rf_daily
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return np.sqrt(TRADING_DAYS) * excess.mean() / std


def annualized_sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS
    excess = returns - rf_daily
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return np.sqrt(TRADING_DAYS) * excess.mean() / downside_std


def max_drawdown(equity_curve: pd.Series) -> float:
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        return np.nan
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return dd.min()


def cagr(equity_curve: pd.Series) -> float:
    equity_curve = pd.Series(equity_curve).dropna()
    if len(equity_curve) < 2:
        return np.nan
    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    n_years = len(equity_curve) / TRADING_DAYS
    if start_val <= 0 or end_val <= 0 or n_years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / n_years) - 1


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    cg = cagr(equity_curve)
    if mdd == 0 or np.isnan(mdd) or np.isnan(cg):
        return np.nan
    return cg / mdd


def avg_turnover(weights_hist: List[np.ndarray]) -> float:
    if len(weights_hist) < 2:
        return np.nan
    turnovers = []
    for prev_w, curr_w in zip(weights_hist[:-1], weights_hist[1:]):
        prev_w = np.asarray(prev_w, dtype=float)
        curr_w = np.asarray(curr_w, dtype=float)
        turnovers.append(np.abs(curr_w - prev_w).sum())
    return float(np.mean(turnovers))


def apply_costs_to_return(gross_ret: float, prev_w: np.ndarray, curr_w: np.ndarray) -> float:
    turnover = float(np.abs(curr_w - prev_w).sum())
    total_cost_rate = turnover * ((TXN_COST_BPS + SLIPPAGE_BPS) / 10000.0)
    return gross_ret - total_cost_rate


def extract_portfolio_value(info: Dict[str, Any], fallback_prev: float, fallback_ret: float) -> float:
    for key in ["portfolio_value", "net_worth", "equity", "account_value"]:
        if key in info:
            return float(info[key])
    return fallback_prev * (1.0 + fallback_ret)


def extract_weights(info: Dict[str, Any], expected_dim: int | None = None) -> np.ndarray:
    candidate_keys = [
        "final_weights",
        "weights",
        "portfolio_weights",
        "stock_weights",
        "action_weights",
    ]
    for key in candidate_keys:
        if key in info:
            arr = np.asarray(info[key], dtype=float).reshape(-1)
            return arr
    if expected_dim is not None:
        return np.zeros(expected_dim, dtype=float)
    return np.array([], dtype=float)


def detect_agent_dim(agent_name: str) -> int:
    if agent_name == "A1":
        return 4
    if agent_name == "A3":
        return 50
    raise ValueError(f"Unknown agent name: {agent_name}")


def detect_daily_return(
    info: Dict[str, Any],
    reward: float,
    prev_equity: float,
    curr_equity: float,
    agent_name: str = "",
) -> float:
    # port_ret is the authoritative daily return key emitted by both envs.
    # After the p9_gym_env.py fix, port_ret is a true fractional return for
    # both A1 and A3 — no rescaling needed.
    for key in ["port_ret", "daily_return", "portfolio_return", "net_return", "raw_return"]:
        if key in info:
            return float(info[key])
    if prev_equity > 0:
        return (curr_equity / prev_equity) - 1.0
    return float(reward)


# ---------------------------------------------------------------------
# OPTIONAL DATA CHECKS
# ---------------------------------------------------------------------

def check_artifacts() -> None:
    # Encoder + data files — always required
    required_files = [ENCODER_CKPT, TEST_PARQUET, X_TEST_TECH, STOCK_META]
    # Every checkpoint in the sweep
    ckpt_paths = [path for (_, path, _, _) in ALL_CHECKPOINTS]
    missing = [str(p) for p in required_files + ckpt_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def alignment_note() -> Dict[str, Any]:
    note = {
        "test_parquet_exists": TEST_PARQUET.exists(),
        "x_test_exists": X_TEST_TECH.exists(),
        "stock_meta_exists": STOCK_META.exists(),
    }
    try:
        test_df = pd.read_parquet(TEST_PARQUET)
        note["test_rows"] = int(len(test_df))
        note["test_start"] = str(test_df.index[0])
        note["test_end"] = str(test_df.index[-1])
    except Exception as e:
        note["test_read_error"] = str(e)

    try:
        x_test = np.load(X_TEST_TECH)
        note["x_test_shape"] = list(x_test.shape)
    except Exception as e:
        note["x_test_read_error"] = str(e)

    return note


# ---------------------------------------------------------------------
# MODEL + ENV
# ---------------------------------------------------------------------

def load_agent(path: Path) -> SAC:
    return SAC.load(str(path), device="auto")


def build_env(agent_name: str, device: str = "cpu"):
    dataset = MidasDataset(split="test")

    encoder = load_encoder(device=device)

    if agent_name == "A1":
        env = MacroAllocatorEnv(dataset=dataset, encoder=encoder, device=device)
        return env, {"class_name": "MacroAllocatorEnv", "split": "test", "device": device}

    if agent_name == "A3":
        env = StockPickerEnv(dataset=dataset, encoder=encoder, device=device)
        return env, {"class_name": "StockPickerEnv", "split": "test", "device": device}

    raise ValueError(f"Unsupported agent name: {agent_name}")


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

def evaluate_one_run(agent_name: str, model: SAC, seed: int) -> Dict[str, Any]:
    env, env_kwargs = build_env(agent_name, device="cpu")
    obs, reset_info = try_seed_env(env, seed)

    action_dim = detect_agent_dim(agent_name)

    equity_curve = []
    daily_returns = []
    weights_hist = []
    rewards = []

    equity = 1.0
    prev_weights = np.zeros(action_dim, dtype=float)

    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = safe_step(env, action)

        raw_weights = extract_weights(info, expected_dim=action_dim)
        if raw_weights.size == action_dim:
            weight_sum = raw_weights.sum()
            if np.isfinite(weight_sum) and weight_sum != 0:
                curr_weights = raw_weights / weight_sum
            else:
                curr_weights = raw_weights
        else:
            curr_weights = prev_weights.copy()

        gross_equity = extract_portfolio_value(info, equity, float(reward))
        gross_ret = detect_daily_return(info, reward, equity, gross_equity, agent_name=agent_name)

        net_ret = apply_costs_to_return(gross_ret, prev_weights, curr_weights)
        # Clip to a conservative daily range (±15%) as a final overflow guard.
        net_ret = float(np.clip(net_ret, -0.15, 0.15))
        if not np.isfinite(net_ret):
            net_ret = 0.0
        equity = equity * (1.0 + net_ret)
        if not np.isfinite(equity) or equity <= 0:
            equity = equity_curve[-1] if equity_curve else 1.0

        daily_returns.append(net_ret)
        equity_curve.append(equity)
        weights_hist.append(curr_weights.copy())
        rewards.append(float(reward))

        prev_weights = curr_weights
        steps += 1

    eq = pd.Series(equity_curve)
    rets = pd.Series(daily_returns)

    out = {
        "agent": agent_name,
        "seed": seed,
        "steps": steps,
        "env_kwargs": json.dumps(env_kwargs),
        "reset_info": json.dumps(reset_info, default=str),
        "mean_reward": float(np.mean(rewards)) if rewards else np.nan,
        "sharpe": annualized_sharpe(rets, RISK_FREE_RATE_ANNUAL),
        "sortino": annualized_sortino(rets, RISK_FREE_RATE_ANNUAL),
        "max_drawdown": max_drawdown(eq),
        "cagr": cagr(eq),
        "calmar": calmar_ratio(eq),
        "avg_turnover": avg_turnover(weights_hist),
        "final_equity": float(eq.iloc[-1]) if len(eq) else np.nan,
    }
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mean_reward",
        "sharpe",
        "sortino",
        "max_drawdown",
        "cagr",
        "calmar",
        "avg_turnover",
        "final_equity",
        "steps",
    ]
    rows = []
    for agent, g in df.groupby("agent"):
        mean_row = {"agent": agent, "seed": "MEAN"}
        std_row = {"agent": agent, "seed": "STD"}
        for c in metric_cols:
            mean_row[c] = pd.to_numeric(g[c], errors="coerce").mean()
            std_row[c] = pd.to_numeric(g[c], errors="coerce").std(ddof=1)
        rows.extend([mean_row, std_row])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    seed_everything(1234)
    check_artifacts()

    note = alignment_note()
    with open(OUT_DIR / "artifact_alignment_note.json", "w", encoding="utf-8") as f:
        json.dump(note, f, indent=2)

    records: List[Dict[str, Any]] = []
    total = len(ALL_CHECKPOINTS)

    for ckpt_idx, (label, ckpt_path, agent_family, timestep) in enumerate(ALL_CHECKPOINTS, 1):
        print(f"\n[{ckpt_idx}/{total}] Loading {label} <- {ckpt_path.name}")
        model = load_agent(ckpt_path)

        for seed in SEEDS:
            seed_everything(seed)
            rec = evaluate_one_run(agent_family, model, seed)
            # Overwrite the generic agent name with the checkpoint-specific label
            rec["agent"] = label
            rec["agent_family"] = agent_family
            rec["checkpoint_name"] = ckpt_path.name
            rec["timestep"] = timestep if timestep is not None else "special"
            records.append(rec)
            print(
                f"  seed={seed:>2}  sharpe={rec['sharpe']:+.3f}  "
                f"cagr={rec['cagr']:+.3f}  mdd={rec['max_drawdown']:+.3f}  "
                f"final_eq={rec['final_equity']:.4f}"
            )

        del model  # free memory between checkpoints

    df = pd.DataFrame(records)
    summary = summarize(df)

    df.to_csv(OUT_DIR / "per_seed_results.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_results.csv", index=False)

    # ── Best checkpoint per agent family ──────────────────────────────
    best_rows = []
    for family in ["A1", "A3"]:
        sub = summary[summary["agent"].str.startswith(family)].copy()
        sub_mean = sub[sub["seed"] == "MEAN"].copy()
        if not sub_mean.empty:
            best_row = sub_mean.loc[sub_mean["sharpe"].idxmax()]
            best_rows.append(best_row)
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUT_DIR / "best_checkpoints.csv", index=False)

    markdown_lines = []
    markdown_lines.append("# P10 Validation — Full Checkpoint Sweep (A1 & A3)")
    markdown_lines.append("")
    markdown_lines.append(f"- Seeds: {SEEDS}")
    markdown_lines.append(f"- Checkpoints evaluated: {total} ({len(A1_CHECKPOINTS)} A1, {len(A3_CHECKPOINTS)} A3)")
    markdown_lines.append(f"- Test parquet: `{TEST_PARQUET}`")
    markdown_lines.append(f"- X_test_technical: `{X_TEST_TECH}`")
    markdown_lines.append(f"- Costs: {TXN_COST_BPS} bps transaction + {SLIPPAGE_BPS} bps slippage")
    markdown_lines.append("")
    markdown_lines.append("## Per-seed results (all checkpoints)")
    markdown_lines.append("")
    markdown_lines.append(df.to_markdown(index=False))
    markdown_lines.append("")
    markdown_lines.append("## Summary (mean ± std across seeds)")
    markdown_lines.append("")
    markdown_lines.append(summary.to_markdown(index=False))
    markdown_lines.append("")
    markdown_lines.append("## Best checkpoint per agent family (by mean Sharpe)")
    markdown_lines.append("")
    markdown_lines.append(best_df.to_markdown(index=False))
    markdown_lines.append("")

    with open(OUT_DIR / "validation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    print("\nSaved:")
    print(OUT_DIR / "per_seed_results.csv")
    print(OUT_DIR / "summary_results.csv")
    print(OUT_DIR / "best_checkpoints.csv")
    print(OUT_DIR / "validation_report.md")
    print(OUT_DIR / "artifact_alignment_note.json")


if __name__ == "__main__":
    main()