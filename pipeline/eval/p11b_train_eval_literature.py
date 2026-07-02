import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
import quantstats as qs

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pipeline.packaging.p9_gym_env import MidasDataset, StockPickerEnv, _MidasEncoder
from models.literature_baselines import HARLFStyleFeaturesExtractor, SAMPHDRLStyleFeaturesExtractor
from pipeline.eval.p10_validate_a1_a3 import seed_everything

CKPT_DIR = Path("checkpoints/baselines")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("results/p11_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_and_eval_baseline(model_name, extractor_class, dataset_train, dataset_test, encoder):
    print(f"\n{'='*50}")
    print(f"Training Baseline: {model_name}")
    print(f"{'='*50}")
    
    # ── Multi-Seed Training & Evaluation ──
    seeds = [7, 11, 19]
    all_sharpes = []
    all_cagrs = []
    all_mdds = []
    all_calmars = []
    
    for seed in seeds:
        seed_everything(seed)
        env_train = StockPickerEnv(dataset=dataset_train, encoder=encoder, device=device)
        
        policy_kwargs = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(features_dim=128),
        )
        
        model = SAC(
            "MlpPolicy",
            env_train,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=128,
            device=device,
            verbose=0,
            seed=seed
        )
        
        # Progress Bar Callback
        from stable_baselines3.common.callbacks import BaseCallback
        from tqdm import tqdm
        
        class TqdmCallback(BaseCallback):
            def __init__(self, total_timesteps: int):
                super().__init__()
                self.pbar = None
                self.total = total_timesteps
                
            def _on_training_start(self) -> None:
                self.pbar = tqdm(total=self.total, desc=f"Seed {seed} Training")
                
            def _on_step(self) -> bool:
                self.pbar.update(1)
                return True
                
            def _on_training_end(self) -> None:
                self.pbar.close()
                
        # Train a new proxy model from scratch for this seed
        model.learn(total_timesteps=50000, callback=TqdmCallback(50000))
        
        # Evaluate
        env_test = StockPickerEnv(dataset=dataset_test, encoder=encoder, device=device)
        obs, _ = env_test.reset(seed=seed)
        done = False
        
        equity_curve = [1.0]
        daily_returns = []
        equity = 1.0
        prev_weights = np.zeros(50, dtype=np.float32)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            exp_a = np.exp(action - action.max())
            w = (exp_a / exp_a.sum()).astype(np.float32)
            
            t = env_test.t
            stock_rets = dataset_test.stock_returns[dataset_test.stock_idx[t]].copy()
            
            gross_ret = float((w * stock_rets).sum())
            turnover = float(np.abs(w - prev_weights).sum())
            net_ret = gross_ret - turnover * 0.0015
            
            equity = equity * (1.0 + net_ret)
            daily_returns.append(net_ret)
            equity_curve.append(equity)
            prev_weights = w
            
            obs, _, done, _, _ = env_test.step(action)
            
        # QuantStats requires a DatetimeIndex for max_drawdown
        dates = pd.date_range(start="2023-01-01", periods=len(daily_returns), freq="B")
        rets = pd.Series(daily_returns, index=dates)
        sharpe = qs.stats.sharpe(rets)
        cagr = qs.stats.cagr(rets)
        mdd = qs.stats.max_drawdown(rets)
        calmar = qs.stats.calmar(rets)
        
        all_sharpes.append(sharpe)
        all_cagrs.append(cagr)
        all_mdds.append(mdd)
        all_calmars.append(calmar)
        
    # Save the last seeded model
    model.save(CKPT_DIR / model_name)
        
    def get_ci(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        margin = 1.96 * (std / np.sqrt(len(arr)))
        return f"{mean:.4f} ± {margin:.4f}"
        
    metrics = {
        "Sharpe Ratio": get_ci(all_sharpes),
        "Max Drawdown": get_ci(all_mdds),
        "CAGR": get_ci(all_cagrs),
        "Calmar Ratio": get_ci(all_calmars)
    }
    print(f"Results for {model_name} (Averaged over seeds {seeds}):")
    print(f"Sharpe: {metrics['Sharpe Ratio']} | CAGR: {metrics['CAGR']} | MDD: {metrics['Max Drawdown']}")
    return metrics

def main():
    print("Loading datasets...")
    # For speed and proper Phase 11 testing, use validation split for proxy training (2021-2022) 
    # instead of the full 2008-2020 which takes hours.
    dataset_train = MidasDataset(split="val")
    dataset_test = MidasDataset(split="test")
    
    print("Loading Unified Encoder...")
    ckpt = torch.load(Path("checkpoints/transformer_encoder.pt"), map_location=device, weights_only=False)
    encoder = _MidasEncoder()
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device).eval()
    
    results = {}
    
    # HARLF
    res_harlf = train_and_eval_baseline("HARLF_style_baseline", HARLFStyleFeaturesExtractor, dataset_train, dataset_test, encoder)
    results["HARLF-style (Iqbal & Ramachandran 2026 proxy)"] = res_harlf
    
    # SAMP-HDRL
    res_samp = train_and_eval_baseline("SAMP_HDRL_style_baseline", SAMPHDRLStyleFeaturesExtractor, dataset_train, dataset_test, encoder)
    results["SAMP-HDRL-style proxy"] = res_samp
    
    # Save results
    df = pd.DataFrame(results).T
    df.to_csv(OUT_DIR / "literature_metrics.csv")
    print("\nLiterature baseline metrics saved to results/p11_baselines/literature_metrics.csv")

if __name__ == "__main__":
    main()
