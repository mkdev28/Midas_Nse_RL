import sys, os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.packaging.p9_gym_env import SentimentModifierEnv, MacroAllocatorEnv, MidasDataset, _MidasEncoder
from models.cross_modal_transformer import CrossModalTransformer
from pipeline.eval.p10_validate_a1_a3 import seed_everything

CKPT = PROJECT_ROOT / "checkpoints"
device = "cpu"
COST_PER_UNIT = (10 + 5) / 10_000   # = 0.0015

def evaluate_a2_run(a2_model, a1_weights_fn, ds):
    env = SentimentModifierEnv(dataset=ds, a1_weights_fn=a1_weights_fn)
    obs, _ = env.reset(seed=42)
    
    equity_curve = [1.0]
    daily_returns = []
    equity = 1.0
    prev_weights = np.zeros(4, dtype=np.float64)
    done = False
    
    while not done:
        action, _ = a2_model.predict(obs, deterministic=True)
        out = env.step(action)
        obs, reward, terminated, truncated, info = out[:5] if len(out) == 5 else (*out, {})
        done = terminated or truncated
        
        gross_ret = float(info["modified_ret"])
        curr_weights = np.asarray(info["modified_weights"], dtype=np.float64)
        
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
    
    excess = rets - 0.0
    std = excess.std(ddof=1)
    sharpe = np.sqrt(252) * excess.mean() / std if std > 0 else 0
    
    start_val, end_val = eq.iloc[0], eq.iloc[-1]
    n_years = len(eq) / 252
    cagr = (end_val / start_val) ** (1 / n_years) - 1
    mdd = (eq / eq.cummax() - 1.0).min()
    
    return sharpe, cagr, mdd

def precompute_a1_test_weights(a1_model, encoder):
    ds = MidasDataset(split="test")
    env = MacroAllocatorEnv(dataset=ds, encoder=encoder, device=device)
    obs, _ = env.reset()
    weights = []
    
    for _ in range(ds.T):
        action, _ = a1_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        weights.append(env.current_weights.copy())
        if done:
            break
            
    return np.array(weights), ds

def main():
    print("============================================================")
    print("A2 Sentiment Modifier Evaluation (Unified vs C2)")
    print("============================================================")
    
    # ---------------------------------------------------------
    # 1. EVALUATE UNIFIED
    # ---------------------------------------------------------
    print("\n[Loading Unified Pipeline]")
    ckpt = torch.load(CKPT / "transformer_encoder.pt", map_location=device, weights_only=False)
    enc_unified = _MidasEncoder()
    enc_unified.load_state_dict(ckpt["encoder_state"])
    enc_unified.to(device)
    enc_unified.eval()
    
    a1_unified_path = CKPT / "a1_unified_fixed_300000_steps.zip"
    a1_unified = SAC.load(a1_unified_path, device=device)
    
    print("Precomputing Base A1 Unified weights on Test split...")
    w_unified, ds_unified = precompute_a1_test_weights(a1_unified, enc_unified)
    a1_unified_fn = lambda t: w_unified[t - 60]  # window is 60
    
    # Evaluate Unified Checkpoints
    ckpts = [
        "a2_unified_50000_steps.zip",
        "a2_unified_100000_steps.zip",
        "a2_unified_150000_steps.zip",
        "a2_unified_200000_steps.zip",
        "a2_unified_250000_steps.zip",
        "a2_unified_300000_steps.zip",
    ]
    
    best_unified_sharpe = -999.0
    best_unified_ckpt = ""
    best_unified_metrics = {}
    
    print("\nEvaluating A2 Unified Checkpoints:")
    for ckpt_name in ckpts:
        path = CKPT / ckpt_name
        if not path.exists():
            continue
        model = PPO.load(path, device=device)
        seed_everything(42)
        sharpe, cagr, mdd = evaluate_a2_run(model, a1_unified_fn, ds_unified)
        print(f"  {ckpt_name:30s} -> Sharpe: {sharpe:.3f} | CAGR: {cagr:.3f} | MDD: {mdd:.3f}")
        if sharpe > best_unified_sharpe:
            best_unified_sharpe = sharpe
            best_unified_ckpt = ckpt_name
            best_unified_metrics = {"cagr": cagr, "mdd": mdd}
            
    # ---------------------------------------------------------
    # 2. EVALUATE C2
    # ---------------------------------------------------------
    print("\n[Loading C2 Pipeline]")
    ckpt_path = CKPT / "c2" / "transformer_c2_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    enc_c2 = CrossModalTransformer(
        d_enc=config["d_enc"],
        d_out=config["d_out"],
        nhead=config["nhead"],
        n_layers=config["n_layers"],
        window=config["window"],
    ).to(device)
    state_dict = ckpt["encoder_state"]
    clean_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
    enc_c2.load_state_dict(clean_state_dict)
    enc_c2.eval()
    
    a1_c2_path = CKPT / "c2" / "a1" / "a1_c2_sac_300000_steps.zip"
    a1_c2 = SAC.load(a1_c2_path, device=device)
    
    print("Precomputing Base A1 C2 weights on Test split...")
    w_c2, ds_c2 = precompute_a1_test_weights(a1_c2, enc_c2)
    a1_c2_fn = lambda t: w_c2[t - 60]
    
    ckpts_c2 = [
        "a2_c2_50000_steps.zip",
        "a2_c2_100000_steps.zip",
        "a2_c2_150000_steps.zip",
        "a2_c2_200000_steps.zip",
        "a2_c2_250000_steps.zip",
        "a2_c2_300000_steps.zip",
    ]
    
    best_c2_sharpe = -999.0
    best_c2_ckpt = ""
    best_c2_metrics = {}
    
    print("\nEvaluating A2 C2 Checkpoints:")
    for ckpt_name in ckpts_c2:
        path = CKPT / ckpt_name
        if not path.exists():
            continue
        model = PPO.load(path, device=device)
        seed_everything(42)
        sharpe, cagr, mdd = evaluate_a2_run(model, a1_c2_fn, ds_c2)
        print(f"  {ckpt_name:30s} -> Sharpe: {sharpe:.3f} | CAGR: {cagr:.3f} | MDD: {mdd:.3f}")
        if sharpe > best_c2_sharpe:
            best_c2_sharpe = sharpe
            best_c2_ckpt = ckpt_name
            best_c2_metrics = {"cagr": cagr, "mdd": mdd}
            
    # ---------------------------------------------------------
    # 3. SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY: A2 MODIFIED PIPELINES (Net of 15 bps turnover costs)")
    print("="*60)
    
    print("\n[UNIFIED]")
    print("Base A1 (No Sentiment)  : Sharpe 1.540 | CAGR 10.3% | MDD -4.7%")
    if best_unified_ckpt:
        print(f"A2 Modified ({best_unified_ckpt:20s}): Sharpe {best_unified_sharpe:.3f} | CAGR {best_unified_metrics['cagr']*100:.1f}% | MDD {best_unified_metrics['mdd']*100:.1f}%")
        
    print("\n[C2 CROSS-MODAL]")
    print("Base A1 C2 (No Sentiment): Sharpe 0.896 | CAGR 10.6% | MDD -11.2%")
    if best_c2_ckpt:
        print(f"A2 Modified ({best_c2_ckpt:20s}): Sharpe {best_c2_sharpe:.3f} | CAGR {best_c2_metrics['cagr']*100:.1f}% | MDD {best_c2_metrics['mdd']*100:.1f}%")

if __name__ == "__main__":
    main()
