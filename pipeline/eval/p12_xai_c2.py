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
from pipeline.packaging.p9_gym_env import make_envs, MacroAllocatorEnv
from pipeline.eval.p12_xai_explainer import EndToEndA1
from models.cross_modal_transformer import CrossModalTransformer

CKPT_DIR = Path("checkpoints")
OUT_DIR = Path("results/p12_xai/c2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading test dataset and C2 models...")
    device = "cpu"
    # We only need dataset from make_envs
    _, _, dataset, _ = make_envs(split="test", device=device)
    
    # Load C2 Encoder
    ckpt_path = CKPT_DIR / "c2" / "transformer_c2_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    encoder = CrossModalTransformer(
        d_enc=config["d_enc"],
        d_out=config["d_out"],
        nhead=config["nhead"],
        n_layers=config["n_layers"],
        window=config["window"],
    ).to(device)
    clean_state_dict = {k.replace("encoder.", ""): v for k, v in ckpt["encoder_state"].items()}
    encoder.load_state_dict(clean_state_dict)
    encoder.eval()
    
    # Create Env for C2
    env_a1 = MacroAllocatorEnv(dataset=dataset, encoder=encoder, device=device)
    
    # Load C2 A1 model
    a1_model = SAC.load(CKPT_DIR / "c2" / "joint" / "a1_c2_joint_final", device=device)
    
    # Target Crash Day (2024-06-04)
    vix_idx = dataset.feature_cols.index("vix_close")
    valid_start = 60
    valid_vix = dataset.features[valid_start:, vix_idx]
    crash_rel_idx = np.argmax(valid_vix)
    target_idx = valid_start + crash_rel_idx
    target_date = dataset.dates[target_idx]
    target_date_str = pd.to_datetime(target_date).date()
    
    target_x = dataset.features[target_idx - 60 : target_idx]
    target_macro = dataset.macro[target_idx]
    
    # Roll forward env
    obs, _ = env_a1.reset()
    for t_step in range(60, target_idx):
        action, _ = a1_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env_a1.step(action)
        if done: break
    target_cw = env_a1.current_weights.copy()
    
    x_tensor = torch.tensor(target_x, dtype=torch.float32).unsqueeze(0).to(device)
    
    # C2 Attention (3T x 3T)
    encoder.eval()
    attn_list = encoder.get_attention_weights(x_tensor)
    last_layer_attn = attn_list[-1] # (1, nhead, 180, 180)
    attn_matrix_np = last_layer_attn[0].mean(dim=0).numpy() # (180, 180)
    
    # C2 SHAP
    wrapper = EndToEndA1(encoder, a1_model.policy, target_macro, target_cw)
    wrapper.eval()
    
    np.random.seed(42)
    bg_indices = np.random.choice(np.arange(valid_start, dataset.T), size=100, replace=False)
    bg_x = np.array([dataset.features[i-60:i] for i in bg_indices])
    bg_tensor = torch.tensor(bg_x, dtype=torch.float32)
    
    explainer = shap.GradientExplainer(wrapper, bg_tensor)
    test_tensor = x_tensor.repeat(5, 1, 1)
    shap_values = explainer.shap_values(test_tensor)
    
    all_shap = np.array(shap_values)
    shap_stocks = all_shap[:, :, :, 0].mean(axis=0)
    shap_aggregated = np.sum(shap_stocks, axis=0)
    
    total_abs_shap = np.sum(np.abs(shap_aggregated))
    if total_abs_shap > 1e-8:
        shap_aggregated = (shap_aggregated / total_abs_shap) * 100
        
    abs_shap = np.abs(shap_aggregated)
    sorted_idx = np.argsort(abs_shap)[-15:]
    
    # Plot C2 SHAP Summary
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(15)
    features_sorted = [dataset.feature_cols[i] for i in sorted_idx]
    shap_sorted = shap_aggregated[sorted_idx]
    colors = ['green' if x > 0 else 'red' for x in shap_sorted]
    plt.barh(y_pos, shap_sorted, color=colors)
    plt.yticks(y_pos, features_sorted)
    plt.xlabel("Relative SHAP Impact on Stock Logits (%)")
    plt.title(f"C2 Ablation: Feature Importance (Crash Day: {target_date_str})")
    for i, v in enumerate(shap_sorted):
        plt.text(v, i, f" {v:+.1f}%", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_c2.png")
    plt.close()
    
    # Plot C2 Attention Map
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn_matrix_np, cmap="viridis", cbar=True)
    plt.title(f"C2 Cross-Modal Attention Map (Crash Day: {target_date_str})\nTokens: [0-59] Macro, [60-119] Tech, [120-179] Sent")
    
    # Add modality dividers
    plt.axvline(60, color='white', linewidth=1)
    plt.axvline(120, color='white', linewidth=1)
    plt.axhline(60, color='white', linewidth=1)
    plt.axhline(120, color='white', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "attention_map_c2.png")
    plt.close()
    
    print("Done! Artifacts saved to results/p12_xai/c2/")

if __name__ == "__main__":
    main()
