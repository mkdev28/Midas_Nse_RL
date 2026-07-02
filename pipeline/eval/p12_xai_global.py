import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import shap
from stable_baselines3 import SAC

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pipeline.packaging.p9_gym_env import make_envs

CKPT_DIR = Path("checkpoints")
OUT_DIR = Path("results/p12_xai/global")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class EndToEndA1Batch(nn.Module):
    """
    Multi-input wrapper so we can pass X, Macro, and CW simultaneously
    and process all target days in one giant vectorized batch.
    """
    def __init__(self, encoder, policy):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        
    def forward(self, x, macro, cw):
        z = self.encoder.encode(x)
        obs = torch.cat([z, macro, cw], dim=1)
        features = self.policy.actor.features_extractor(obs)
        latent_pi = self.policy.actor.latent_pi(features)
        mean_logits = self.policy.actor.mu(latent_pi)
        
        # 1. Scalar output for SHAP (only care about Stocks logit)
        # SHAP's internal PyTorch implementation expects a 2D output (B, num_classes)
        # so we must return (B, 1) instead of (B,) to prevent index errors.
        stock_logit = mean_logits[:, 0:1]
        return stock_logit

def main():
    print("Loading test dataset and models...")
    device = "cpu"
    env_a1, _, dataset, encoder = make_envs(split="test", device=device)
    a1_model = SAC.load(CKPT_DIR / "a1_unified_joint_final", device=device)
    
    valid_start = 60
    valid_indices = np.arange(valid_start, dataset.T)
    
    np.random.seed(42)
    sample_indices = np.sort(np.random.choice(valid_indices, size=200, replace=False))
    bg_indices     = np.sort(np.random.choice(valid_indices, size=50,  replace=False))
    
    print("Rolling forward environment to capture true A1 weights...")
    all_indices = np.unique(np.concatenate([bg_indices, sample_indices]))
    cw_map = {}
    obs, _ = env_a1.reset()
    t = 60
    for idx in all_indices:
        while t < idx:
            action, _ = a1_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_a1.step(action)
            t += 1
            if done: break
        cw_map[idx] = env_a1.current_weights.copy()

    bg_cw     = torch.from_numpy(np.array([cw_map[i] for i in bg_indices], dtype=np.float32))
    target_cw = torch.from_numpy(np.array([cw_map[i] for i in sample_indices], dtype=np.float32))
    
    def extract_windows(features, indices, window=60):
        idx_matrix = indices[:, None] - np.arange(window, 0, -1)  # (N, window)
        assert idx_matrix.min() >= 0, "Index out of bounds"
        assert idx_matrix.max() < dataset.T, "Index out of bounds"
        return torch.from_numpy(features[idx_matrix].astype(np.float32))

    bg_x     = extract_windows(dataset.features, bg_indices)
    target_x = extract_windows(dataset.features, sample_indices)

    bg_macro     = torch.from_numpy(dataset.macro[bg_indices].astype(np.float32))
    target_macro = torch.from_numpy(dataset.macro[sample_indices].astype(np.float32))
    
    # Build wrapper
    wrapper = EndToEndA1Batch(encoder, a1_model.policy)
    wrapper.eval()
    
    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(wrapper, [bg_x, bg_macro, bg_cw])
    
    print("Computing Batched SHAP for all 200 days simultaneously...")
    # 2. Simplified multi-input SHAP indexing
    shap_vals = explainer.shap_values([target_x, target_macro, target_cw])
    
    # SHAP output format varies based on if it considers (B, 1) single or multi-output
    # Robust extraction of the first input (target_x):
    if isinstance(shap_vals, list) and isinstance(shap_vals[0], list):
        shap_x = shap_vals[0][0]  # list of classes -> list of inputs -> target_x
    elif isinstance(shap_vals, list):
        shap_x = shap_vals[0]     # list of inputs -> target_x
    else:
        shap_x = shap_vals        # directly target_x
        
    if len(shap_x.shape) == 4 and shap_x.shape[2] == 1:
        shap_x = shap_x[:, :, 0, :] # Squeeze (200, 60, 1, 46) -> (200, 60, 46)
        
    # Aggregate over time
    shap_aggregated = np.sum(shap_x, axis=1) # (200, 46)
    
    # Vectorized Normalization
    abs_sums = np.abs(shap_aggregated).sum(axis=1, keepdims=True)
    abs_sums = np.where(abs_sums < 1e-8, 1.0, abs_sums)
    shap_aggregated = (shap_aggregated / abs_sums) * 100
            
    print("Plotting Global Profile...")
    vix_idx = dataset.feature_cols.index("vix_close")
    vix_values = dataset.features[sample_indices, vix_idx]
    
    p75 = np.percentile(vix_values, 75)
    p25 = np.percentile(vix_values, 25)
    
    stress_idx = np.where(vix_values >= p75)[0]
    calm_idx = np.where(vix_values <= p25)[0]
    
    stress_shap = np.asarray(np.mean(np.abs(shap_aggregated[stress_idx]), axis=0))
    calm_shap = np.asarray(np.mean(np.abs(shap_aggregated[calm_idx]), axis=0))
    
    # Flatten strictly before argsort to prevent SHAP's trailing dummy class dim (46, 1) from destroying the sort
    global_mean_shap = np.asarray(np.mean(np.abs(shap_aggregated), axis=0)).flatten()
    
    # Ensure 1-D int array for indexing
    top_10_arr = np.argsort(global_mean_shap)[-10:]
    top_10 = [int(x) for x in top_10_arr]
    
    # Force flat pure-float lists to prevent Matplotlib array shape crashes
    stress_shap_flat = np.array(stress_shap).flatten()
    calm_shap_flat = np.array(calm_shap).flatten()
    stress_bars = [float(stress_shap_flat[i]) for i in top_10]
    calm_bars = [float(calm_shap_flat[i]) for i in top_10]
    
    plt.figure(figsize=(12, 6))
    features = [dataset.feature_cols[i] for i in top_10]
    
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, stress_bars, width, label='High Stress (Top 25% VIX)', color='firebrick')
    plt.bar(x + width/2, calm_bars, width, label='Calm (Bottom 25% VIX)', color='steelblue')
    
    plt.ylabel('Average Absolute Relative SHAP Impact (%)')
    plt.title('Global Feature Importance Profile (Stress vs Calm Regimes)')
    plt.xticks(x, features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_feature_profile.png")
    
    print("Done! Artifacts saved to results/p12_xai/global/")

if __name__ == "__main__":
    main()
