import os
import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO, SAC

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pipeline.inference.live_data_fetcher import fetch_live_raw_data, compute_live_features, NIFTY_SYMBOLS
from pipeline.packaging.p9_gym_env import _MidasEncoder, MacroAllocatorEnv, StockPickerEnv, SentimentModifierEnv
from pipeline.eval.p11_baselines import get_full_weights, softmax

device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("checkpoints")

class LiveDataset:
    """
    A duck-typed mock of MidasDataset for a single live inference step.
    Wraps the (60, 46) macro, (60, 50, 12) stock, and (60, 3) sentiment windows.
    """
    def __init__(self, X_stock, X_macro, X_sentiment):
        # The env expects t to step. We want inference at t=60.
        self.T = 61
        
        # encoder(x) takes self.features[t - 60 : t]. At t=60, it takes 0:60.
        self.features = np.zeros((61, 46), dtype=np.float32)
        self.features[0:60] = X_macro
        
        # A1 takes self.macro[t]. We'll use the last 5 dims of macro.
        self.macro = np.zeros((61, 5), dtype=np.float32)
        self.macro[60] = X_macro[-1, :5]
        
        # A3 takes self.X_stock[self.stock_idx[t]]. 
        self.X_stock = np.zeros((1, 50, 12), dtype=np.float32)
        self.X_stock[0] = X_stock[-1] # The latest day's features
        
        self.stock_idx = np.zeros(61, dtype=np.int32)
        self.stock_idx[60] = 0
        
        # A2 takes self.sentiment[t].
        self.sentiment = np.zeros((61, 3), dtype=np.float32)
        self.sentiment[60] = X_sentiment[-1]
        
        # Env step uses self.returns[t]. We mock it to 0.
        self.returns = np.zeros(61, dtype=np.float32)

def run_live_inference():
    print("="*60)
    print(" MIDAS-NSE LIVE INFERENCE PIPELINE")
    print("="*60)
    
    # 1. Fetch live data
    today = datetime.today()
    print(f"Fetching Live Data for {today.strftime('%Y-%m-%d')}...", flush=True)
    prices, macro, news = fetch_live_raw_data(NIFTY_SYMBOLS, today, window_days=100)
    X_stock, X_macro, X_sentiment = compute_live_features(prices, macro, news, today, window_size=60)
    
    # 2. Build Mock Dataset and Load Encoder
    live_dataset = LiveDataset(X_stock, X_macro, X_sentiment)
    
    print("Loading Unified Transformer Encoder...", flush=True)
    ckpt = torch.load(CKPT_DIR / "transformer_encoder.pt", map_location=device, weights_only=False)
    encoder = _MidasEncoder().to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    
    # 3. Instantiate Gym Environments to get EXACT obs
    print("Initializing Simulation Environments...", flush=True)
    a1_env = MacroAllocatorEnv(dataset=live_dataset, encoder=encoder, device=device)
    a3_env = StockPickerEnv(dataset=live_dataset, encoder=encoder, device=device)
    a2_env = SentimentModifierEnv(dataset=live_dataset, a1_weights_fn=lambda t: a1_env.current_weights)
    
    # Fast-forward to t=60 (live inference step)
    a1_env.t = 60
    a3_env.t = 60
    a2_env.t = 60
    
    # _get_obs() automatically calls the encoder and constructs exact training shapes
    obs_a1 = a1_env._get_obs()
    obs_a3 = a3_env._get_obs()
    obs_a2 = a2_env._get_obs()
    
    print(f"Generated Strict Observations: A1 {obs_a1.shape} | A2 {obs_a2.shape} | A3 {obs_a3.shape}")
    
    # 4. Load Agents
    print("Loading RL Agents...", flush=True)
    a1 = SAC.load(CKPT_DIR / "a1_unified_joint_final", device=device)
    a2 = PPO.load(CKPT_DIR / "a2_unified_joint_final", device="cpu")
    a3 = SAC.load(CKPT_DIR / "a3_unified_joint_final", device=device)
    
    # 5. Predict
    act_a1, _ = a1.predict(obs_a1, deterministic=True)
    w_a1 = softmax(act_a1).astype(np.float32)
    a1_env.current_weights = w_a1 # Important for A2 to see the chosen weights
    
    act_a3, _ = a3.predict(obs_a3, deterministic=True)
    w_a3 = softmax(act_a3).astype(np.float32)
    
    # Recompute A2 obs so it sees the new w_a1
    obs_a2 = a2_env._get_obs()
    act_a2, _ = a2.predict(obs_a2, deterministic=True)
    w_a2 = np.clip(act_a2, 0.5, 1.5).astype(np.float32)
    
    # 6. Coordinate
    w_full = get_full_weights(w_a1, w_a2, w_a3)
    
    # Output to Console
    print("\n[Target Portfolio Allocations]")
    print(f"Bonds:     {w_a1[1]:.2%}")
    print(f"Commodity: {w_a1[2]:.2%}")
    print(f"Cash:      {w_a1[3]:.2%}")
    print("Stocks:")
    
    # Only print stocks with > 0.5% allocation
    # w_full represents the relative weighting *within* the stock bucket. 
    # To get absolute portfolio allocation, multiply by w_a1[0]
    allocations = []
    full_53_weights = []
    for sym, weight in zip(NIFTY_SYMBOLS, w_full):
        absolute_weight = weight * w_a1[0]
        full_53_weights.append(absolute_weight)
        if absolute_weight > 0.005:
            allocations.append((sym, absolute_weight))
            
    # Add macro weights to complete the 53-dim vector
    full_53_weights.extend([w_a1[1], w_a1[2], w_a1[3]])
    
    # 7. Sanity Check
    total_weight = sum(full_53_weights)
    print(f"\n[Sanity Check] Total Portfolio Weight: {total_weight:.4f}")
    if not np.isclose(total_weight, 1.0, atol=1e-3):
        print("WARNING: Total weight does not sum to 1.0!")
        
    # 8. Save CSV
    import pandas as pd
    out_path = Path(f"results/live_allocations_{today.strftime('%Y-%m-%d')}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_symbols = NIFTY_SYMBOLS + ["BONDS", "COMMODITIES", "CASH"]
    df_weights = pd.DataFrame({"Symbol": all_symbols, "Weight": full_53_weights})
    df_weights.to_csv(out_path, index=False)
    print(f"Saved full 53-dim allocations to: {out_path}")
            
    # 9. Simple Log Line
    allocations.sort(key=lambda x: x[1], reverse=True)
    for sym, w in allocations:
        print(f"  - {sym:15s}: {w:.2%}")
        
    print("\n--- LIVE PIPELINE SUMMARY LOG ---")
    print(f"Date: {today.strftime('%Y-%m-%d')}")
    print(f"A1 Class Weights (Stocks, Bonds, Cmdty, Cash): {np.round(w_a1, 3)}")
    print(f"A2 Sentiment Mods (Bonds, Cmdty, Cash): {np.round(w_a2, 3)}")
    top_10 = [(s, round(w, 4)) for s, w in allocations[:10]]
    print(f"A3 Top 10 Stocks: {top_10}")
        
    print("\nLive Inference Complete.")

if __name__ == "__main__":
    run_live_inference()
