import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from pipeline.packaging.p9_gym_env import MacroAllocatorEnv, MidasDataset
from models.cross_modal_transformer import CrossModalTransformer
from pipeline.eval.p10_validate_a1_a3 import evaluate_one_run, seed_everything

def main():
    print("============================================================")
    print("A1 C2 Architecture — Final Evaluation")
    print("============================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load C2 Encoder
    ckpt_path = PROJECT_ROOT / "checkpoints" / "c2" / "transformer_c2_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = ckpt["config"]
    encoder = CrossModalTransformer(
        d_enc=config["d_enc"],
        d_out=config["d_out"],
        nhead=config["nhead"],
        n_layers=config["n_layers"],
        window=config["window"],
    ).to(device)
    
    state_dict = ckpt["encoder_state"]
    clean_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(clean_state_dict)
    encoder.eval()
    
    # 2. Setup Dataset & Env
    dataset = MidasDataset(split="test")
    env = MacroAllocatorEnv(dataset, encoder, device)
    
    # 3. Load trained SAC agent
    agent_path = PROJECT_ROOT / "checkpoints" / "c2" / "a1" / "a1_c2_sac_300000_steps.zip"
    if not agent_path.exists():
        print(f"ERROR: Checkpoint not found at {agent_path}")
        return
        
    model = SAC.load(str(agent_path), device=device)
    
    seeds = [7, 11, 19]
    ckpts = [
        "a1_c2_sac_50000_steps.zip",
        "a1_c2_sac_100000_steps.zip",
        "a1_c2_sac_150000_steps.zip",
        "a1_c2_sac_200000_steps.zip",
        "a1_c2_sac_250000_steps.zip",
        "a1_c2_sac_300000_steps.zip",
    ]
    
    COST_PER_UNIT = (10 + 5) / 10_000   # = 0.0015
    best_sharpe = -999.0
    best_ckpt = ""
    
    print("\nRunning Test Split Evaluation (with 15 bps x turnover transaction costs)...")
    
    for ckpt_name in ckpts:
        agent_path = PROJECT_ROOT / "checkpoints" / "c2" / "a1" / ckpt_name
        if not agent_path.exists():
            continue
            
        print(f"\nEvaluating {ckpt_name}...")
        model = SAC.load(str(agent_path), device=device)
        
        sharpes, cagrs, mdds = [], [], []
        
        for seed in seeds:
            seed_everything(seed)
            obs, _ = env.reset(seed=seed)
            equity_curve, daily_returns = [1.0], []
            equity = 1.0
            prev_weights = np.zeros(4, dtype=np.float64)
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                out = env.step(action)
                obs, reward, terminated, truncated, info = out[:5] if len(out) == 5 else (*out, {})
                done = terminated or truncated
                
                gross_ret = float(info.get('port_ret', reward))
                curr_weights = np.asarray(info.get("weights", prev_weights), dtype=np.float64)
                
                turnover = float(np.abs(curr_weights - prev_weights).sum())
                cost = turnover * COST_PER_UNIT
                net_ret = float(np.clip(gross_ret - cost, -0.15, 0.15))
                if not np.isfinite(net_ret):
                    net_ret = 0.0
                    
                equity = equity * (1.0 + net_ret)
                daily_returns.append(net_ret)
                equity_curve.append(equity)
                prev_weights = curr_weights
                
            rets = pd.Series(daily_returns)
            eq = pd.Series(equity_curve)
            
            excess = rets - 0.0 # Rf = 0
            std = excess.std(ddof=1)
            sharpe = np.sqrt(252) * excess.mean() / std if std > 0 else 0
            
            start_val, end_val = eq.iloc[0], eq.iloc[-1]
            n_years = len(eq) / 252
            cagr = (end_val / start_val) ** (1 / n_years) - 1
            mdd = (eq / eq.cummax() - 1.0).min()
            
            sharpes.append(sharpe)
            cagrs.append(cagr)
            mdds.append(mdd)
            print(f"  Seed {seed:2d} -> Sharpe: {sharpe:.3f} | CAGR: {cagr:.3f} | MDD: {mdd:.3f}")
            
        mean_sharpe = np.mean(sharpes)
        print(f"  --> MEAN -> Sharpe: {mean_sharpe:.3f} | CAGR: {np.mean(cagrs):.3f} | MDD: {np.mean(mdds):.3f}")
        
        if mean_sharpe > best_sharpe:
            best_sharpe = mean_sharpe
            best_ckpt = ckpt_name
            
        del model
        
    print("=" * 60)
    print(f"BEST CHECKPOINT: {best_ckpt} with Mean Sharpe: {best_sharpe:.3f}")
    print("=" * 60)
    
    print("\n[Baseline Targets to Beat]")
    print("A1 Unified (Best) : Sharpe 1.540 | CAGR 10.3% | MDD -4.7%")
    print("Equal Weight P11  : Sharpe 1.876 | CAGR 23.6% | MDD -15.8%")

if __name__ == "__main__":
    main()
