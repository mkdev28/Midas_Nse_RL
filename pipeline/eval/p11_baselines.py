import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC, PPO
import quantstats as qs

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pipeline.packaging.p9_gym_env import (
    make_envs, MidasDataset, _MidasEncoder,
    MacroAllocatorEnv, SentimentModifierEnv, StockPickerEnv, coordinate
)
from models.cross_modal_transformer import CrossModalTransformer
from pipeline.eval.p13_latency_profiler import load_c2_encoder

CKPT_DIR = Path("checkpoints")
OUT_DIR = Path("results/p11_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_full_weights(w_a1, w_a2, w_a3):
    # Temporarily mimic p10's coordinate to see if the issue is macro drag
    modified = w_a1.copy()
    modified[:3] *= w_a2
    modified = modified / modified.sum()
    final = modified[0] * w_a3
    return (final / final.sum()).astype(np.float32)

def main():
    print("Loading test dataset...", flush=True)
    dataset = MidasDataset(split="test")
    
    # ── Load Unified Models ──────────────────────────────────────────────
    print("Loading Unified Architecture Models...", flush=True)
    ckpt = torch.load(CKPT_DIR / "transformer_encoder.pt", map_location=device, weights_only=False)
    encoder_u = _MidasEncoder()
    encoder_u.load_state_dict(ckpt["encoder_state"])
    encoder_u.to(device).eval()
    
    a1_u = SAC.load(CKPT_DIR / "a1_unified_joint_final", device=device)
    a2_u = PPO.load(CKPT_DIR / "a2_unified_joint_final", device="cpu")
    a3_u = SAC.load(CKPT_DIR / "a3_unified_joint_final", device=device)
    
    # ── Load C2 Models ──────────────────────────────────────────────
    print("Loading C2 Architecture Models...", flush=True)
    encoder_c2 = load_c2_encoder(device=device)
    
    a1_c2 = SAC.load(CKPT_DIR / "c2" / "joint" / "a1_c2_joint_final", device=device)
    a2_c2 = PPO.load(CKPT_DIR / "c2" / "joint" / "a2_c2_joint_final", device="cpu")
    a3_c2 = SAC.load(CKPT_DIR / "c2" / "joint" / "a3_c2_joint_final", device=device)

    seed = 7
    print(f"\n--- Running Deterministic Evaluation (Seed {seed}) ---", flush=True)
    # Instantiate Environments for this seed
    a1_env_u = MacroAllocatorEnv(dataset=dataset, encoder=encoder_u, device=device)
    a3_env_u = StockPickerEnv(dataset=dataset, encoder=encoder_u, device=device)
    a2_env_u = SentimentModifierEnv(dataset=dataset, a1_weights_fn=lambda t: a1_env_u.current_weights)
    
    obs_a1_u, _ = a1_env_u.reset(seed=seed)
    obs_a3_u, _ = a3_env_u.reset(seed=seed)
    obs_a2_u, _ = a2_env_u.reset(seed=seed)

    a1_env_c2 = MacroAllocatorEnv(dataset=dataset, encoder=encoder_c2, device=device)
    a3_env_c2 = StockPickerEnv(dataset=dataset, encoder=encoder_c2, device=device)
    a2_env_c2 = SentimentModifierEnv(dataset=dataset, a1_weights_fn=lambda t: a1_env_c2.current_weights)
    
    obs_a1_c2, _ = a1_env_c2.reset(seed=seed)
    obs_a3_c2, _ = a3_env_c2.reset(seed=seed)
    obs_a2_c2, _ = a2_env_c2.reset(seed=seed)
    
    returns_dict = {
        "MIDAS-NSE (Unified baseline)": [],
        "MIDAS-NSE (C2 Production)": [],
        "Flat RL (A3 Only)": [],
        "Equal Weight (Stocks)": [],
        "NIFTY 50": []
    }
    
    prev_w_u = np.zeros(50, dtype=np.float32)
    prev_w_c2 = np.zeros(50, dtype=np.float32)
    prev_w_flat = np.zeros(50, dtype=np.float32)
    prev_w_eq = np.zeros(50, dtype=np.float32)
    
    dates = []
    done_u = False
    done_c2 = False
    
    while not done_u and not done_c2:
        t = a1_env_u.t
        dt = np.datetime64('2023-01-02') + np.timedelta64(dataset.stock_idx[t] - dataset.stock_idx[0], 'D')
        dates.append(dt)
        
        stock_rets = dataset.stock_returns[dataset.stock_idx[t]].copy()
        asset_rets = np.array([
            dataset.returns[t],
            dataset.bond_ret[t],
            dataset.commodity_ret[t],
            dataset.cash_ret[t]
        ])
        
        # 1. NIFTY 50
        returns_dict["NIFTY 50"].append(asset_rets[0])
        
        # 2. Equal Weight
        w_eq_stock = np.ones(50) / 50.0
        gross_eq = np.sum(w_eq_stock * stock_rets)
        to_eq = np.sum(np.abs(w_eq_stock - prev_w_eq))
        prev_w_eq = w_eq_stock
        returns_dict["Equal Weight (Stocks)"].append(gross_eq - 0.0015 * to_eq)

        # 3. Unified Architecture
        act_a1_u, _ = a1_u.predict(obs_a1_u, deterministic=True)
        act_a3_u, _ = a3_u.predict(obs_a3_u, deterministic=True)
        act_a2_u, _ = a2_u.predict(obs_a2_u, deterministic=True)
        
        w_a1_u = softmax(act_a1_u).astype(np.float32)
        w_a3_u = softmax(act_a3_u).astype(np.float32)
        w_a2_u = np.clip(act_a2_u, 0.5, 1.5).astype(np.float32)
        w_full_u = get_full_weights(w_a1_u, w_a2_u, w_a3_u)
        
        gross_u = float((w_full_u * stock_rets).sum())
        to_u = np.sum(np.abs(w_full_u - prev_w_u))
        prev_w_u = w_full_u
        returns_dict["MIDAS-NSE (Unified baseline)"].append(gross_u - 0.0015 * to_u)

        # 4. C2 Ablation
        act_a1_c2, _ = a1_c2.predict(obs_a1_c2, deterministic=True)
        act_a3_c2, _ = a3_c2.predict(obs_a3_c2, deterministic=True)
        act_a2_c2, _ = a2_c2.predict(obs_a2_c2, deterministic=True)
        
        w_a1_c2 = softmax(act_a1_c2).astype(np.float32)
        w_a3_c2 = softmax(act_a3_c2).astype(np.float32)
        w_a2_c2 = np.clip(act_a2_c2, 0.5, 1.5).astype(np.float32)
        w_full_c2 = get_full_weights(w_a1_c2, w_a2_c2, w_a3_c2)
        
        gross_c2 = float((w_full_c2 * stock_rets).sum())
        to_c2 = np.sum(np.abs(w_full_c2 - prev_w_c2))
        prev_w_c2 = w_full_c2
        returns_dict["MIDAS-NSE (C2 Production)"].append(gross_c2 - 0.0015 * to_c2)

        # 5. Flat RL (A3 Only)
        w_a1_flat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        w_a2_flat = np.ones(3, dtype=np.float32)
        w_full_flat = get_full_weights(w_a1_flat, w_a2_flat, w_a3_u)
        
        gross_flat = float((w_full_flat * stock_rets).sum())
        to_flat = np.sum(np.abs(w_full_flat - prev_w_flat))
        prev_w_flat = w_full_flat
        returns_dict["Flat RL (A3 Only)"].append(gross_flat - 0.0015 * to_flat)

        # Advance Environments
        _, _, done_u, _, _ = a1_env_u.step(act_a1_u)
        _, _, _, _, _ = a2_env_u.step(act_a2_u)
        _, _, _, _, _ = a3_env_u.step(act_a3_u)
        
        _, _, done_c2, _, _ = a1_env_c2.step(act_a1_c2)
        _, _, _, _, _ = a2_env_c2.step(act_a2_c2)
        _, _, _, _, _ = a3_env_c2.step(act_a3_c2)
        
        if not done_u:
            obs_a1_u = a1_env_u._get_obs()
            obs_a2_u = a2_env_u._get_obs()
            obs_a3_u = a3_env_u._get_obs()
            
            obs_a1_c2 = a1_env_c2._get_obs()
            obs_a2_c2 = a2_env_c2._get_obs()
            obs_a3_c2 = a3_env_c2._get_obs()

    df = pd.DataFrame(returns_dict, index=dates)
    metrics_data = {}
    for col in df.columns:
        metrics_data[col] = {
            "Sharpe Ratio": qs.stats.sharpe(df[col]),
            "Max Drawdown": qs.stats.max_drawdown(df[col]),
                "CAGR": qs.stats.cagr(df[col]),
                "Calmar Ratio": qs.stats.calmar(df[col])
            }
        
        qs.reports.html(df["MIDAS-NSE (C2 Production)"], benchmark=df["NIFTY 50"], output=str(OUT_DIR / "c2_vs_nifty.html"), title="C2 Production vs NIFTY 50")
        df.to_csv(OUT_DIR / "daily_returns.csv")
    
    print("Generating metrics_deterministic.csv...", flush=True)
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_df.to_csv(OUT_DIR / "metrics_deterministic.csv")
    print("Done! Tearsheets and metrics_deterministic.csv saved in results/p11_baselines/", flush=True)

if __name__ == "__main__":
    main()
