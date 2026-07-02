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
OUT_DIR = Path("results/p12_xai/regimes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class EndToEndA1(nn.Module):
    def __init__(self, encoder, policy, target_macro, target_weights):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        self.register_buffer('macro', torch.tensor(target_macro, dtype=torch.float32))
        self.register_buffer('cw', torch.tensor(target_weights, dtype=torch.float32))
        
    def forward(self, x):
        z = self.encoder.encode(x)
        B = x.size(0)
        m = self.macro.unsqueeze(0).expand(B, -1)
        c = self.cw.unsqueeze(0).expand(B, -1)
        obs = torch.cat([z, m, c], dim=1)
        features = self.policy.actor.features_extractor(obs)
        latent_pi = self.policy.actor.latent_pi(features)
        mean_actions = self.policy.actor.mu(latent_pi)
        # Return logits
        return mean_actions

def get_attention_matrix(encoder, x_tensor):
    x = encoder.input_proj(x_tensor)
    x = encoder.pos_enc(x)
    for layer in encoder.encoder.layers[:-1]:
        x = layer(x)
    last_layer = encoder.encoder.layers[-1]
    attn_output, attn_weights = last_layer.self_attn(x, x, x, need_weights=True)
    return attn_weights

def analyze_regime(regime_name, target_idx, env_a1, dataset, a1_model, encoder, device, bg_tensor):
    print(f"\n--- Analyzing Regime: {regime_name} (Index: {target_idx}) ---")
    target_date = dataset.dates[target_idx]
    vix_idx = dataset.feature_cols.index("vix_close")
    target_vix = dataset.features[target_idx, vix_idx]
    target_ret = dataset.features[target_idx, dataset.feature_cols.index("ret_1d")]
    
    print(f"Targeting {regime_name} day: {target_date} (VIX = {target_vix:.2f}, Ret_1d = {target_ret:.2f})")
    
    regime_dir = OUT_DIR / regime_name
    regime_dir.mkdir(exist_ok=True)
    
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
    
    # Attention
    encoder.eval()
    with torch.no_grad():
        attn_matrix = get_attention_matrix(encoder, x_tensor)
    attn_matrix_np = attn_matrix[0].cpu().numpy()
    
    # SHAP
    wrapper = EndToEndA1(encoder, a1_model.policy, target_macro, target_cw)
    wrapper.eval()
    
    explainer = shap.GradientExplainer(wrapper, bg_tensor)
    test_tensor = x_tensor.repeat(5, 1, 1)
    shap_values = explainer.shap_values(test_tensor)
    
    all_shap = np.array(shap_values)
    shap_stocks = all_shap[:, :, :, 0].mean(axis=0)
    shap_aggregated = np.sum(shap_stocks, axis=0)
    
    total_abs_shap = np.sum(np.abs(shap_aggregated))
    if total_abs_shap > 1e-8:
        shap_aggregated = (shap_aggregated / total_abs_shap) * 100
        shap_stocks = (shap_stocks / total_abs_shap) * 100
        
    abs_shap = np.abs(shap_aggregated)
    sorted_idx = np.argsort(abs_shap)[-15:]
    top2_idx = sorted_idx[-2:]
    
    # Generate Plots
    target_date_str = pd.to_datetime(target_date).date()
    
    # 1. SHAP Summary
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(15)
    features_sorted = [dataset.feature_cols[i] for i in sorted_idx]
    shap_sorted = shap_aggregated[sorted_idx]
    colors = ['green' if x > 0 else 'red' for x in shap_sorted]
    plt.barh(y_pos, shap_sorted, color=colors)
    plt.yticks(y_pos, features_sorted)
    plt.xlabel("Relative SHAP Impact on Stock Logits (%)")
    plt.title(f"{regime_name.capitalize()} Regime: Feature Importance\nDay: {target_date_str}")
    for i, v in enumerate(shap_sorted):
        plt.text(v, i, f" {v:+.1f}%", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(regime_dir / "shap_summary.png")
    plt.close()
    
    # Find Attention Hotspots (mean attention per column)
    col_attn = attn_matrix_np.mean(axis=0)
    hotspots = np.argsort(col_attn)[-2:] # Top 2 attended days
    
    # 2. Temporal SHAP with Attention Overlay
    plt.figure(figsize=(10, 4))
    days = np.arange(-60, 0)
    for i in top2_idx:
        feat_name = dataset.feature_cols[i]
        plt.plot(days, shap_stocks[:, i], label=feat_name, marker='o', markersize=4)
    
    for hotspot in hotspots:
        rel_hotspot = hotspot - 60
        plt.axvspan(rel_hotspot-0.5, rel_hotspot+0.5, color='yellow', alpha=0.3, label=f'Attn Hotspot (t{rel_hotspot})')
        
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("Historical Days (0 is today)")
    plt.ylabel("Relative SHAP Impact (%)")
    plt.title(f"{regime_name.capitalize()} Regime: Temporal SHAP vs Attention\nDay: {target_date_str}")
    
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(regime_dir / "temporal_shap.png")
    plt.close()
    
    # 3. Attention Map
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix_np, cmap="viridis")
    plt.title(f"{regime_name.capitalize()} Regime: Attention Map\nDay: {target_date_str}")
    plt.xlabel("Key (Historical Days: 0 is t-60, 59 is t-1)")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.savefig(regime_dir / "attention_map.png")
    plt.close()
    
    cf_markdown = ""
    # Counterfactual Perturbation (only on crash)
    if regime_name == "crash":
        print("Running Counterfactual Perturbations...")
        cf_markdown += "## Counterfactual Sanity Checks\n"
        cf_markdown += "| Feature | Perturbation | Target Date Value | Expected SHAP Impact | Actual $\Delta$ Logit |\n"
        cf_markdown += "|---------|--------------|-------------------|----------------------|-----------------------|\n"
        
        # Get baseline logit
        with torch.no_grad():
            base_logit = wrapper(x_tensor)[0, 0].item()
            
        for i in top2_idx:
            feat_name = dataset.feature_cols[i]
            orig_val = target_x[-1, i] # Perturb t-1 (which is index 59)
            std_val = dataset.features[:, i].std()
            shap_sign = np.sign(shap_aggregated[i])
            
            for direction, dir_name in [(1, "+1 STD"), (-1, "-1 STD")]:
                perturbed_x = target_x.copy()
                perturbed_x[-1, i] += direction * std_val
                p_tensor = torch.tensor(perturbed_x, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    new_logit = wrapper(p_tensor)[0, 0].item()
                delta = new_logit - base_logit
                expected = "Positive" if (shap_sign * direction) > 0 else "Negative"
                actual = "Positive" if delta > 0 else "Negative"
                cf_markdown += f"| {feat_name} ({dir_name}) | {orig_val:.2f} $\\to$ {perturbed_x[-1, i]:.2f} | {expected} | {actual} ($\Delta {delta:+.3f}$) |\n"

    # Report writing
    top3_pos = [dataset.feature_cols[i] for i in np.argsort(shap_aggregated)[-3:] if shap_aggregated[i] > 0]
    top3_neg = [dataset.feature_cols[i] for i in np.argsort(shap_aggregated)[:3] if shap_aggregated[i] < 0]
    
    report_content = f"""# XAI Report: {regime_name.capitalize()} Regime

- **Target Day**: {target_date_str}
- **VIX**: {target_vix:.2f}
- **Return (1d)**: {(target_ret*100):.2f}%
- **A1 Stock Weight**: {(target_cw[0]*100):.1f}%

### Top Contributors
- **Top 3 Positive Drivers**: {', '.join(top3_pos)}
- **Top 3 Negative Drivers**: {', '.join(top3_neg)}

![SHAP Summary](shap_summary.png)
![Temporal SHAP](temporal_shap.png)
![Attention Map](attention_map.png)

{cf_markdown}
"""
    with open(regime_dir / f"report_{regime_name}.md", "w") as f:
        f.write(report_content)

def main():
    device = "cpu"
    env_a1, _, dataset, encoder = make_envs(split="test", device=device)
    a1_model = SAC.load(CKPT_DIR / "a1_unified_joint_final", device=device)
    
    valid_start = 60
    valid_vix = dataset.features[valid_start:, dataset.feature_cols.index("vix_close")]
    valid_ret = dataset.features[valid_start:, dataset.feature_cols.index("ret_1d")]
    
    # Heuristics
    crash_rel_idx = np.argmax(valid_vix)
    crash_idx = valid_start + crash_rel_idx
    
    # Calm: VIX <= 40th percentile, abs return <= 0.25%, min abs return
    vix_p40 = np.percentile(valid_vix, 40)
    calm_candidates = np.where((valid_vix <= vix_p40) & (np.abs(valid_ret) <= 0.0025))[0]
    if len(calm_candidates) == 0:
        calm_candidates = np.where(valid_vix <= vix_p40)[0] # fallback
    calm_rel_idx = calm_candidates[np.argmin(np.abs(valid_ret[calm_candidates]))]
    calm_idx = valid_start + calm_rel_idx
    
    # Rally: VIX between 40-70th percentile, max return
    vix_p70 = np.percentile(valid_vix, 70)
    rally_candidates = np.where((valid_vix > vix_p40) & (valid_vix <= vix_p70))[0]
    rally_rel_idx = rally_candidates[np.argmax(valid_ret[rally_candidates])]
    rally_idx = valid_start + rally_rel_idx
    
    # Background
    np.random.seed(42)
    bg_indices = np.random.choice(np.arange(valid_start, dataset.T), size=100, replace=False)
    bg_x = np.array([dataset.features[i-60:i] for i in bg_indices])
    bg_tensor = torch.tensor(bg_x, dtype=torch.float32)
    
    regimes = [
        ("crash", crash_idx),
        ("calm", calm_idx),
        ("rally", rally_idx)
    ]
    
    for r_name, r_idx in regimes:
        analyze_regime(r_name, r_idx, env_a1, dataset, a1_model, encoder, device, bg_tensor)

if __name__ == "__main__":
    main()
