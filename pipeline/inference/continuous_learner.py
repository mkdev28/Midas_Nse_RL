"""
MIDAS-NSE — Continuous Learner (Shadow A3 Learning)
====================================================
Implements safe online adaptation of the A3 (Stock Picker) agent via shadow learning.

ARCHITECTURE:
  "Shadow A/B Learning" — two parallel A3 models run simultaneously:
    - Production A3: immutable. Backed by full P10/P11 offline evaluation.
    - Shadow A3:     adapts daily from live experience. Never auto-promoted.

THREE SAFETY CONSTRAINTS (paper-grade):
  1. Immutable production — production checkpoints NEVER modified automatically.
  2. Sharpe safety gate — shadow saved only if rolling 30-day Sharpe is within
     SAFETY_MARGIN of production. Gate requires MIN_HISTORY_DAYS=30 to activate.
  3. Bounded daily gradient budget — max DAILY_GRAD_STEPS per run.
     Reduced to 1 step if shadow is already underperforming (additional protection).

PRODUCTION PROMOTION:
  Shadow → Production is MANUAL via p12_promote_shadow_to_production.py,
  which runs a full offline backtest on 6–12 months of test data before
  asking for explicit user approval.

USAGE (run once per trading day after market close):
  python pipeline/inference/continuous_learner.py [--date YYYY-MM-DD]

PAPER SECTION: "Live Deployment — Safe Online Adaptation via Shadow A3 Learning"
"""

import sys
import os
import pickle
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from pipeline.inference.live_data_fetcher import (
    fetch_live_raw_data, compute_live_features, NIFTY_SYMBOLS
)
from pipeline.packaging.p9_gym_env import _MidasEncoder, StockPickerEnv
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
DAILY_GRAD_STEPS       = 3      # Max gradient steps/day under normal conditions
REDUCED_GRAD_STEPS     = 1      # Steps/day when shadow is underperforming gate
SAFETY_MARGIN          = 0.05   # Allowed underperformance: shadow >= prod - 0.05
ROLLING_WINDOW         = 30     # Days for rolling Sharpe/MDD computation (gate uses this)
MIN_HISTORY_DAYS       = 30     # Gate only activates after this many days (= ROLLING_WINDOW)
TAIL_RISK_VIX_THRESH   = 25.0   # VIX threshold for tail-risk tagging (mirrors C5)
TAIL_RISK_FII_ZSCORE   = 2.0    # FII z-score threshold for tail-risk tagging (mirrors C5)
TAIL_RISK_KEEP         = 50     # Max tail-risk episodes to permanently retain in shadow buffer
SHADOW_BUFFER_SIZE     = 500    # Total shadow replay buffer capacity
SHADOW_BATCH_SIZE      = 32     # Batch size for shadow gradient steps
SHADOW_LR              = 1e-5   # 10× lower than A3 standalone (1e-4)
TX_COST_BPS            = 0.0015 # 15 bps — matches offline training exactly

# ── Paths ──────────────────────────────────────────────────────────────────────
CKPT_DIR         = Path("checkpoints")
PROC_DIR         = Path("data/processed")
PROD_CKPT        = CKPT_DIR / "a3_unified_joint_final"   # SB3 loads both .zip and dir
LIVE_DIR         = CKPT_DIR / "live"
SHADOW_DIR       = LIVE_DIR / "shadow_model"
PROD_COPY_DIR    = LIVE_DIR / "prod_baseline"
SHADOW_CKPT      = SHADOW_DIR / "a3_shadow"
PROD_COPY_CKPT   = PROD_COPY_DIR / "a3_prod_copy"
SHADOW_LOG       = LIVE_DIR / "shadow_log.csv"
ROLLING_LOG      = LIVE_DIR / "rolling_returns.csv"
LAST_STATE_PKL   = LIVE_DIR / "last_state.pkl"           # Persists (obs, weights, prev_weights)
TAIL_RISK_PKL    = LIVE_DIR / "tail_risk_episodes.pkl"   # Persisted tail-risk buffer

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


def _load_encoder() -> _MidasEncoder:
    """Load production Transformer encoder (frozen)."""
    ckpt = torch.load(CKPT_DIR / "transformer_encoder.pt",
                      map_location=device, weights_only=False)
    encoder = _MidasEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    return encoder


def _fii_zscore(fii_net_value: float, rolling_log: pd.DataFrame) -> float:
    """Compute FII z-score against 30-day rolling history from log."""
    if "fii_net_30d_mean" not in rolling_log.columns or len(rolling_log) < 5:
        return 0.0
    mu  = rolling_log["fii_net_30d_mean"].tail(30).mean()
    sig = rolling_log["fii_net_30d_mean"].tail(30).std()
    if sig < 1e-8:
        return 0.0
    return float((fii_net_value - mu) / sig)


def _is_tail_risk(vix: float, fii_zscore: float) -> bool:
    """Mirror C5 SelectiveReplayBuffer's tail-risk condition exactly."""
    return (vix > TAIL_RISK_VIX_THRESH) or (abs(fii_zscore) > TAIL_RISK_FII_ZSCORE)


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup_shadow_model() -> bool:
    """
    First-time setup: copy production A3 to shadow and prod_baseline dirs.
    Returns True if setup was performed, False if shadow already exists.

    INVARIANT: After this function, production A3 at PROD_CKPT is NEVER touched again.
    """
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    PROD_COPY_DIR.mkdir(parents=True, exist_ok=True)

    prod_zip    = Path(str(PROD_CKPT)      + ".zip")
    shadow_zip  = Path(str(SHADOW_CKPT)    + ".zip")
    ref_zip     = Path(str(PROD_COPY_CKPT) + ".zip")

    if not prod_zip.exists():
        raise FileNotFoundError(
            f"Production checkpoint not found: {prod_zip}\n"
            "Complete the joint fine-tuning pipeline first."
        )

    first_run = not shadow_zip.exists()
    if first_run:
        shutil.copy2(str(prod_zip), str(shadow_zip))
        print(f"[CL] Shadow model initialized from production: {shadow_zip}")

    if not ref_zip.exists():
        shutil.copy2(str(prod_zip), str(ref_zip))
        print(f"[CL] Immutable production reference saved: {ref_zip}")

    # Initialize log files
    if not SHADOW_LOG.exists():
        pd.DataFrame(columns=[
            "date", "shadow_sharpe", "prod_sharpe", "sharpe_delta",
            "shadow_mdd_30d", "prod_mdd_30d",
            "shadow_turnover_30d", "prod_turnover_30d",
            "avg_vix_30d", "avg_fii_zscore_30d",
            "is_tail_risk_day", "action_taken",
            "history_days", "grad_steps"
        ]).to_csv(SHADOW_LOG, index=False)

    if not ROLLING_LOG.exists():
        pd.DataFrame(columns=[
            "date", "shadow_ret", "prod_ret",
            "shadow_turnover", "prod_turnover",
            "shadow_weights_hash", "prod_weights_hash",
            "vix", "fii_zscore"
        ]).to_csv(ROLLING_LOG, index=False)

    if first_run:
        print("[CL] First-time setup complete. Running in observation mode until "
              f"{MIN_HISTORY_DAYS} days of history accumulated.")
    return first_run


# ── State persistence ──────────────────────────────────────────────────────────

def save_state(obs_a3, shadow_weights, prod_weights, prev_shadow_weights, prev_prod_weights):
    """Persist today's state for tomorrow's experience construction."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "obs_a3":              obs_a3,
        "shadow_weights":      shadow_weights,
        "prod_weights":        prod_weights,
        "prev_shadow_weights": prev_shadow_weights,
        "prev_prod_weights":   prev_prod_weights,
    }
    with open(LAST_STATE_PKL, "wb") as f:
        pickle.dump(state, f)


def load_state():
    """
    Load yesterday's state. Returns dict or None on first run.
    Keys: obs_a3, shadow_weights, prod_weights, prev_shadow_weights, prev_prod_weights
    """
    if not LAST_STATE_PKL.exists():
        return None
    with open(LAST_STATE_PKL, "rb") as f:
        return pickle.load(f)


# ── Reward computation ─────────────────────────────────────────────────────────

def compute_portfolio_return(
    weights: np.ndarray,          # (50,) portfolio weights
    prev_weights: np.ndarray,     # (50,) previous day's weights
    actual_returns: np.ndarray    # (50,) today's true fractional returns
) -> tuple[float, float]:
    """
    Compute net portfolio return and turnover.
    Returns (net_return, turnover) — matches offline training formula exactly.

    R_net = Σ(w_i × r_i) − TX_COST_BPS × Σ|w_i − w_prev_i|
    """
    ret_clean = np.nan_to_num(actual_returns, nan=0.0, posinf=0.0, neginf=0.0)
    gross_ret = float(np.dot(weights, ret_clean))
    turnover  = float(np.abs(weights - prev_weights).sum())
    net_ret   = gross_ret - TX_COST_BPS * turnover
    return float(np.clip(net_ret, -0.2, 0.2)), turnover


# ── C5-aligned tail-risk buffer ────────────────────────────────────────────────

def load_tail_risk_buffer() -> list:
    """Load persisted tail-risk episodes (across sessions)."""
    if TAIL_RISK_PKL.exists():
        with open(TAIL_RISK_PKL, "rb") as f:
            return pickle.load(f)
    return []


def save_tail_risk_buffer(episodes: list):
    with open(TAIL_RISK_PKL, "wb") as f:
        pickle.dump(episodes, f)


def update_tail_risk_buffer(episodes: list, new_episode: dict) -> list:
    """
    Add a new tail-risk episode. Cap at TAIL_RISK_KEEP (FIFO on overflow).
    Mirrors C5 SelectiveReplayBuffer logic.
    """
    episodes.append(new_episode)
    if len(episodes) > TAIL_RISK_KEEP:
        episodes = episodes[-TAIL_RISK_KEEP:]   # discard oldest tail-risk episode
    return episodes


# ── Shadow gradient update ─────────────────────────────────────────────────────

def shadow_gradient_step(
    shadow_model: SAC,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_obs: np.ndarray,
    tail_risk_episodes: list,
    n_steps: int,
):
    """
    Add today's experience to shadow buffer + optionally inject a tail-risk episode,
    then run n_steps gradient updates.

    C5 philosophy: tail-risk episodes are re-injected to prevent forgetting.
    """
    # Add today's experience
    shadow_model.replay_buffer.add(
        obs.reshape(1, -1),
        next_obs.reshape(1, -1),
        action.reshape(1, -1),
        np.array([[reward]], dtype=np.float32),
        np.array([[False]]),
        [{}]
    )

    # Inject one random tail-risk episode if buffer exists (C5 philosophy)
    if tail_risk_episodes:
        ep = tail_risk_episodes[np.random.randint(len(tail_risk_episodes))]
        shadow_model.replay_buffer.add(
            ep["obs"].reshape(1, -1),
            ep["next_obs"].reshape(1, -1),
            ep["action"].reshape(1, -1),
            np.array([[ep["reward"]]], dtype=np.float32),
            np.array([[False]]),
            [{}]
        )

    buf_size = shadow_model.replay_buffer.size()
    if buf_size >= SHADOW_BATCH_SIZE:
        shadow_model.train(gradient_steps=n_steps, batch_size=SHADOW_BATCH_SIZE)
        print(f"[CL] Shadow model: {n_steps} gradient step(s). Buffer: {buf_size}")
    else:
        print(f"[CL] Buffer too small ({buf_size} < {SHADOW_BATCH_SIZE}). Skipping gradient update.")


# ── Rolling statistics ─────────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = ROLLING_WINDOW):
    """Annualized Sharpe over last `window` days. None if insufficient data."""
    r = returns.dropna().tail(window)
    if len(r) < window:
        return None
    std = r.std()
    return float(r.mean() / std * np.sqrt(252)) if std > 1e-8 else 0.0


def rolling_mdd(returns: pd.Series, window: int = ROLLING_WINDOW) -> float:
    """Max drawdown over last `window` days."""
    r = returns.dropna().tail(window)
    if len(r) < 5:
        return 0.0
    equity = (1 + r).cumprod()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    return float(dd.min())


def rolling_avg_turnover(turnover_col: pd.Series, window: int = ROLLING_WINDOW) -> float:
    return float(turnover_col.dropna().tail(window).mean())


# ── Safety gate ────────────────────────────────────────────────────────────────

def safety_gate(shadow_sharpe_val, prod_sharpe_val) -> bool:
    """
    Allow shadow save only if Sharpe degradation < SAFETY_MARGIN.

    SAFETY_MARGIN = 0.05 means:
      shadow can underperform production by at most 0.05 Sharpe.
      Any greater underperformance → shadow discarded, previous shadow retained.

    Gate returns False if history is insufficient (None Sharpe).
    Gate returns True if production Sharpe is also None (both still building history).
    """
    if shadow_sharpe_val is None:
        return False   # Not enough shadow history
    if prod_sharpe_val is None:
        return True    # No production baseline yet; allow shadow to build
    return shadow_sharpe_val >= (prod_sharpe_val - SAFETY_MARGIN)


# ── Logging ────────────────────────────────────────────────────────────────────

def log_decision(date_str, shadow_sh, prod_sh, shadow_mdd, prod_mdd,
                 shadow_to, prod_to, avg_vix, avg_fii_z, is_tail_risk,
                 action_taken, history_days, grad_steps):
    row = {
        "date":                date_str,
        "shadow_sharpe":       round(shadow_sh,  4) if shadow_sh  is not None else None,
        "prod_sharpe":         round(prod_sh,    4) if prod_sh    is not None else None,
        "sharpe_delta":        round(shadow_sh - prod_sh, 4)
                               if (shadow_sh is not None and prod_sh is not None) else None,
        "shadow_mdd_30d":      round(shadow_mdd, 4),
        "prod_mdd_30d":        round(prod_mdd,   4),
        "shadow_turnover_30d": round(shadow_to,  4),
        "prod_turnover_30d":   round(prod_to,    4),
        "avg_vix_30d":         round(avg_vix,    2),
        "avg_fii_zscore_30d":  round(avg_fii_z,  3),
        "is_tail_risk_day":    is_tail_risk,
        "action_taken":        action_taken,
        "history_days":        history_days,
        "grad_steps":          grad_steps,
    }
    pd.DataFrame([row]).to_csv(SHADOW_LOG, mode="a", header=False, index=False)


def log_returns(date_str, shadow_ret, prod_ret, shadow_to, prod_to,
                shadow_w_hash, prod_w_hash, vix, fii_zscore):
    # NOTE: column order MUST match header written in setup_shadow_model()
    row = {
        "date":                date_str,
        "shadow_ret":          round(shadow_ret, 6),
        "prod_ret":            round(prod_ret,   6),
        "shadow_turnover":     round(shadow_to,  4),   # turnover before hash
        "prod_turnover":       round(prod_to,    4),
        "shadow_weights_hash": shadow_w_hash,
        "prod_weights_hash":   prod_w_hash,
        "vix":                 round(vix,        3),
        "fii_zscore":          round(fii_zscore, 3),
    }
    pd.DataFrame([row]).to_csv(ROLLING_LOG, mode="a", header=False, index=False)


# ── Main orchestration ─────────────────────────────────────────────────────────

def run_continuous_learner(date: datetime = None):
    """
    Main entry point. Run once per trading day after market close.

    Day T lifecycle:
      1. Load yesterday's state (obs, shadow_weights, prod_weights)
      2. Fetch today's actual stock returns
      3. Compute SEPARATE rewards for shadow and production (same costs, different weights)
      4. Run gradient steps on shadow model (budgeted)
      5. Compute rolling statistics for both models
      6. Safety gate → save or discard shadow update
      7. Log everything for paper narrative
      8. Save today's state for tomorrow
    """
    if date is None:
        date = datetime.today()
    today_str = date.strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"  MIDAS-NSE SHADOW A3 LEARNER - {today_str}")
    print("=" * 60)

    # ── Step 1: Setup ──────────────────────────────────────────────────────
    setup_shadow_model()
    tail_risk_episodes = load_tail_risk_buffer()

    # ── Step 2: Load yesterday's state ────────────────────────────────────
    state = load_state()
    if state is None:
        print("[CL] No saved state from yesterday. First run — generating obs only.")
        obs_yesterday         = None
        shadow_weights_yest   = None
        prod_weights_yest     = None
        prev_shadow_weights   = np.full(50, 1/50, dtype=np.float32)
        prev_prod_weights     = np.full(50, 1/50, dtype=np.float32)
    else:
        obs_yesterday         = state["obs_a3"]
        shadow_weights_yest   = state["shadow_weights"]
        prod_weights_yest     = state["prod_weights"]
        prev_shadow_weights   = state["prev_shadow_weights"]
        prev_prod_weights     = state["prev_prod_weights"]
        print(f"[CL] Loaded yesterday's state. "
              f"Shadow weights hash: {hash(shadow_weights_yest.tobytes()) & 0xFFFF:04X}, "
              f"Prod weights hash: {hash(prod_weights_yest.tobytes()) & 0xFFFF:04X}")

    # ── Step 3: Fetch today's live data ───────────────────────────────────
    print(f"[CL] Fetching today's data ({today_str})...")
    try:
        prices, macro, news = fetch_live_raw_data(NIFTY_SYMBOLS, date, window_days=100)
        X_stock, X_macro, X_sentiment = compute_live_features(
            prices, macro, news, date, window_size=60
        )
    except Exception as e:
        print(f"[CL] ERROR fetching live data: {e}")
        print("[CL] Skipping today. State not updated.")
        return

    # De-normalize actual stock returns for today
    try:
        means        = np.load(PROC_DIR / "stock_features_means.npy")   # (1,1,12)
        stds         = np.load(PROC_DIR / "stock_features_stds.npy")    # (1,1,12)
        ret_mean     = float(means.flat[9])
        ret_std      = float(stds.flat[9])
        z_returns    = X_stock[-1, :, 9]                                  # (50,) z-norm
        actual_ret   = z_returns * ret_std + ret_mean                     # true frac returns
        actual_ret   = np.nan_to_num(actual_ret, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"[CL] ERROR computing returns: {e}")
        return

    # Extract VIX directly from raw macro DataFrame (reliable — no column index guessing)
    # macro is the raw downloaded macro DataFrame from fetch_live_raw_data
    vix_today = 20.0   # safe default
    try:
        vix_col_candidates = [c for c in macro.columns if "VIX" in c.upper() or "vix" in c.lower()]
        if vix_col_candidates:
            vix_series = macro[vix_col_candidates[0]].dropna()
            if not vix_series.empty:
                vix_today = float(vix_series.iloc[-1])
    except Exception:
        pass

    df_rolling  = pd.read_csv(ROLLING_LOG, parse_dates=["date"]) if ROLLING_LOG.exists() else pd.DataFrame()
    fii_z_today = _fii_zscore(0.0, df_rolling)
    is_tail     = _is_tail_risk(vix_today, fii_z_today)

    print(f"[CL] Today VIX~{vix_today:.1f} | Tail-risk day: {is_tail}")
    print(f"[CL] Actual stock returns: mean={actual_ret.mean():.4f}, std={actual_ret.std():.4f}")

    # ── Step 4: Build today's A3 observations for BOTH models ─────────────
    encoder      = _load_encoder()
    from pipeline.inference.live_inference import LiveDataset

    live_ds      = LiveDataset(X_stock, X_macro, X_sentiment)
    a3_env       = StockPickerEnv(dataset=live_ds, encoder=encoder, device=device)
    a3_env.t     = 60
    obs_today    = a3_env._get_obs()

    # Get today's shadow and production actions
    shadow_model = SAC.load(str(SHADOW_CKPT), device=device,
                            custom_objects={"learning_rate": SHADOW_LR,
                                            "buffer_size":   SHADOW_BUFFER_SIZE})
    shadow_model.learning_rate = SHADOW_LR

    prod_model   = SAC.load(str(PROD_CKPT), device=device)

    act_shadow, _ = shadow_model.predict(obs_today, deterministic=True)
    act_prod,   _ = prod_model.predict(obs_today,   deterministic=True)

    shadow_weights_today = _softmax(act_shadow)
    prod_weights_today   = _softmax(act_prod)

    # ── Step 5: Compute SEPARATE rewards for shadow and production ─────────
    # This is the key fix: shadow and production use their OWN weights,
    # NOT the same value. This makes the Sharpe comparison meaningful.
    grad_steps_done = 0
    shadow_ret_today = 0.0
    prod_ret_today   = 0.0
    shadow_to_today  = 0.0
    prod_to_today    = 0.0

    if obs_yesterday is not None:
        shadow_ret_today, shadow_to_today = compute_portfolio_return(
            shadow_weights_yest, prev_shadow_weights, actual_ret
        )
        prod_ret_today, prod_to_today = compute_portfolio_return(
            prod_weights_yest, prev_prod_weights, actual_ret
        )
        print(f"[CL] Yesterday's returns -> Shadow: {shadow_ret_today*100:.2f}% "
              f"(to={shadow_to_today:.3f}) | Prod: {prod_ret_today*100:.2f}% "
              f"(to={prod_to_today:.3f})")

        # ── Step 6: Gradient update ────────────────────────────────────────
        # Determine step budget based on current gate status
        if ROLLING_LOG.exists() and len(df_rolling) >= MIN_HISTORY_DAYS:
            sh_sh = rolling_sharpe(df_rolling["shadow_ret"])
            pr_sh = rolling_sharpe(df_rolling["prod_ret"])
            underperforming = (sh_sh is not None and pr_sh is not None and
                               sh_sh < pr_sh - SAFETY_MARGIN)
            n_steps = REDUCED_GRAD_STEPS if underperforming else DAILY_GRAD_STEPS
            if underperforming:
                print(f"[CL] Shadow underperforming gate — reducing gradient steps to {n_steps}.")
        else:
            n_steps = DAILY_GRAD_STEPS

        shadow_gradient_step(
            shadow_model=shadow_model,
            obs=obs_yesterday,
            action=shadow_weights_yest,
            reward=shadow_ret_today,
            next_obs=obs_today,
            tail_risk_episodes=tail_risk_episodes,
            n_steps=n_steps,
        )
        grad_steps_done = n_steps

        # ── Step 7: Tail-risk episode retention (C5 alignment) ─────────────
        if is_tail:
            episode = {
                "obs":     obs_yesterday,
                "action":  shadow_weights_yest,
                "reward":  shadow_ret_today,
                "next_obs": obs_today,
                "vix":     vix_today,
                "date":    today_str,
            }
            tail_risk_episodes = update_tail_risk_buffer(tail_risk_episodes, episode)
            save_tail_risk_buffer(tail_risk_episodes)
            print(f"[CL] Tail-risk episode retained. Total stored: {len(tail_risk_episodes)}")

    # ── Step 8: Log daily returns ──────────────────────────────────────────
    if obs_yesterday is not None:
        log_returns(
            today_str,
            shadow_ret_today, prod_ret_today,
            shadow_to_today, prod_to_today,
            hash(shadow_weights_yest.tobytes()) & 0xFFFF,
            hash(prod_weights_yest.tobytes())   & 0xFFFF,
            vix_today, fii_z_today
        )

    # ── Step 9: Rolling statistics ─────────────────────────────────────────
    df_rolling = pd.read_csv(ROLLING_LOG, parse_dates=["date"]) if ROLLING_LOG.exists() else pd.DataFrame()
    history_days = len(df_rolling)

    shadow_sh = rolling_sharpe(df_rolling["shadow_ret"])   if history_days > 0 else None
    prod_sh   = rolling_sharpe(df_rolling["prod_ret"])     if history_days > 0 else None
    shadow_mdd_val = rolling_mdd(df_rolling["shadow_ret"]) if history_days > 0 else 0.0
    prod_mdd_val   = rolling_mdd(df_rolling["prod_ret"])   if history_days > 0 else 0.0
    shadow_to_avg  = rolling_avg_turnover(df_rolling["shadow_turnover"]) if "shadow_turnover" in df_rolling else 0.0
    prod_to_avg    = rolling_avg_turnover(df_rolling["prod_turnover"])   if "prod_turnover"   in df_rolling else 0.0
    avg_vix_30d    = float(df_rolling["vix"].tail(ROLLING_WINDOW).mean()) if "vix" in df_rolling else vix_today
    avg_fii_30d    = float(df_rolling["fii_zscore"].tail(ROLLING_WINDOW).mean()) if "fii_zscore" in df_rolling else 0.0

    # ── Step 10: Safety gate and save decision ─────────────────────────────
    action_taken = "NO_UPDATE"

    if grad_steps_done > 0:
        if history_days < MIN_HISTORY_DAYS:
            # Building history phase — save shadow (no gate yet)
            shadow_model.save(str(SHADOW_CKPT))
            action_taken = f"SAVED_BUILDING_HISTORY ({history_days}/{MIN_HISTORY_DAYS})"
            print(f"[CL] Building history ({history_days}/{MIN_HISTORY_DAYS}d). Shadow saved.")

        elif safety_gate(shadow_sh, prod_sh):
            shadow_model.save(str(SHADOW_CKPT))
            delta = (shadow_sh - prod_sh) if (shadow_sh and prod_sh) else 0
            action_taken = f"SAVED_GATE_PASSED (Δ={delta:+.3f})"
            print(f"[CL] ✅ Safety gate PASSED. Shadow saved. "
                  f"Shadow Sharpe={shadow_sh:.3f} | Prod={prod_sh:.3f} | Δ={delta:+.3f}")
        else:
            action_taken = "DISCARDED_GATE_FAILED"
            delta = (shadow_sh - prod_sh) if (shadow_sh and prod_sh) else 0
            print(f"[CL] ❌ Safety gate FAILED. Shadow update discarded.")
            print(f"     Shadow Sharpe={shadow_sh:.3f} < Prod={prod_sh:.3f} − {SAFETY_MARGIN}")
            print(f"     Δ={delta:+.3f} < −{SAFETY_MARGIN}. Previous shadow checkpoint retained.")

    # ── Step 11: Log the decision ──────────────────────────────────────────
    log_decision(
        today_str, shadow_sh, prod_sh, shadow_mdd_val, prod_mdd_val,
        shadow_to_avg, prod_to_avg, avg_vix_30d, avg_fii_30d,
        is_tail, action_taken, history_days, grad_steps_done
    )

    # ── Step 12: Save today's state for tomorrow ───────────────────────────
    save_state(
        obs_today,
        shadow_weights_today,
        prod_weights_today,
        prev_shadow_weights=shadow_weights_yest if shadow_weights_yest is not None
                            else np.full(50, 1/50, dtype=np.float32),
        prev_prod_weights=prod_weights_yest if prod_weights_yest is not None
                          else np.full(50, 1/50, dtype=np.float32),
    )

    print(f"\n[CL] State saved for tomorrow.")
    print(f"[CL] Shadow log: {SHADOW_LOG}")
    print(f"[CL] History: {history_days} days | Tail-risk episodes stored: {len(tail_risk_episodes)}")
    print("[CL] Shadow A3 learning cycle complete.")
    print()
    print("NOTE: To promote shadow -> production, run:")
    print("  python pipeline/inference/p12_promote_shadow_to_production.py")
    print("This runs a full offline backtest before asking for your approval.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MIDAS-NSE Shadow A3 Learner — safe online adaptation"
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Date to run for (YYYY-MM-DD). Defaults to today.")
    args = parser.parse_args()
    run_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.today()
    run_continuous_learner(date=run_date)
