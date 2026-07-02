import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
from pipeline.packaging.p9_gym_env import (
    MacroAllocatorEnv, SentimentModifierEnv, StockPickerEnv,
    MidasDataset, _MidasEncoder, coordinate
)

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT = Path("checkpoints")
LOGS = Path("logs/joint_unified")
LOGS.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Joint Fine-Tuning on : {device}")

# ── 1. Load Unified Encoder ───────────────────────────────────────────────────
ckpt = torch.load(CKPT / "transformer_encoder.pt", map_location=device, weights_only=False)
encoder = _MidasEncoder()
encoder.load_state_dict(ckpt["encoder_state"])
encoder.to(device)
encoder.eval()
print("Encoder loaded and FROZEN for joint fine-tuning.")

# ── 2. Load the 3 Agents ──────────────────────────────────────────────────────
print("Loading Unified Agents...")
a1_model = SAC.load(CKPT / "a1_unified_fixed_300000_steps.zip", device=device)
a2_model = PPO.load(CKPT / "a2_unified_50000_steps.zip", device="cpu")  # PPO MlpPolicy faster on CPU
a3_model = SAC.load(CKPT / "a3_sac_50000_steps.zip", device=device)

FINETUNE_LR = 1e-5
for model in [a1_model, a3_model]:
    for param_group in model.actor.optimizer.param_groups:
        param_group["lr"] = FINETUNE_LR
    for param_group in model.critic.optimizer.param_groups:
        param_group["lr"] = FINETUNE_LR

for param_group in a2_model.policy.optimizer.param_groups:
    param_group["lr"] = FINETUNE_LR

# Initialize loggers — required when calling .train() manually on a loaded model
a1_model.set_logger(configure(str(LOGS / "a1"), ["stdout"]))
a3_model.set_logger(configure(str(LOGS / "a3"), ["stdout"]))
a2_model.set_logger(configure(str(LOGS / "a2"), ["stdout"]))
print("Loggers initialized.")

# ── 3. Build Datasets & Base Envs ─────────────────────────────────────────────
train_ds = MidasDataset(split="train")
a1_env = MacroAllocatorEnv(dataset=train_ds, encoder=encoder, device=device)
a3_env = StockPickerEnv(dataset=train_ds, encoder=encoder, device=device)
a2_env = SentimentModifierEnv(dataset=train_ds, a1_weights_fn=lambda t: a1_env.current_weights)

# ── 4. Custom Joint Training Loop ─────────────────────────────────────────────
TOTAL_STEPS = 200_000
batch_size = 256
CKPT_INTERVAL = 50_000  # Save intermediate checkpoints to compare overfitting

obs_a1, _ = a1_env.reset()
obs_a3, _ = a3_env.reset()
obs_a2, _ = a2_env.reset()

print(f"Starting Custom Joint Fine-Tuning Loop ({TOTAL_STEPS} steps)...")

portfolio_returns = []
COST_PER_UNIT = (10 + 5) / 10_000  # 15 bps turnover cost
prev_weights = np.zeros(50, dtype=np.float32)

episode_start_a2 = True

for step in range(1, TOTAL_STEPS + 1):
    # Predict actions (with exploration)
    action_a1, _ = a1_model.predict(obs_a1, deterministic=False)
    action_a3, _ = a3_model.predict(obs_a3, deterministic=False)
    
    # PPO requires action, value, and log_prob for rollout buffer
    # SB3 ActorCriticPolicy.forward() returns (actions, values, log_prob)
    # values and log_prob MUST remain as torch.Tensor — rollout_buffer.add() calls .clone() on them
    with torch.no_grad():
        obs_tensor_a2 = torch.tensor(obs_a2).unsqueeze(0).to("cpu")  # A2 is on CPU
        action_a2_t, values_a2_t, log_prob_a2_t = a2_model.policy.forward(obs_tensor_a2)
        action_a2 = action_a2_t.cpu().numpy()[0]   # shape (3,) — numpy for coordinator
        # Keep as tensors for rollout_buffer.add()
        
    # Get physical weights
    exp_a1 = np.exp(action_a1 - action_a1.max())
    w_a1 = (exp_a1 / exp_a1.sum()).astype(np.float32)
    
    exp_a3 = np.exp(action_a3 - action_a3.max())
    w_a3 = (exp_a3 / exp_a3.sum()).astype(np.float32)
    
    w_a2 = np.clip(action_a2, 0.5, 1.5).astype(np.float32)
    
    # ── Coordinator ──
    final_stock_weights = coordinate(w_a1, w_a2, w_a3)
    
    # Compute environment step with transaction costs
    t = a1_env.t
    true_stock_returns = train_ds.stock_returns[train_ds.stock_idx[t]].copy()
    
    gross_ret = float((final_stock_weights * true_stock_returns).sum())
    turnover = float(np.abs(final_stock_weights - prev_weights).sum())
    cost = turnover * COST_PER_UNIT
    net_ret = float(np.clip(gross_ret - cost, -0.15, 0.15))
    
    prev_weights = final_stock_weights
    portfolio_returns.append(net_ret)
    
    # Compute rolling Sharpe for reward
    joint_reward = net_ret
    if len(portfolio_returns) >= 60:
        roll = np.array(portfolio_returns[-60:])
        if roll.std() > 1e-6:
            joint_reward = (roll.mean() / roll.std()) * np.sqrt(252)
            
    # Advance state
    a1_env.t += 1
    a2_env.t += 1
    a3_env.t += 1
    
    done = a1_env.t >= train_ds.T - 1
    
    next_obs_a1 = a1_env._get_obs()
    next_obs_a3 = a3_env._get_obs()
    next_obs_a2 = a2_env._get_obs()
    
    # Store in buffers
    a1_model.replay_buffer.add(obs_a1, next_obs_a1, action_a1, joint_reward, done, [{}])
    a3_model.replay_buffer.add(obs_a3, next_obs_a3, action_a3, joint_reward, done, [{}])
    a2_model.rollout_buffer.add(obs_a2, action_a2, joint_reward, [episode_start_a2], values_a2_t, log_prob_a2_t)
    
    episode_start_a2 = False
    
    obs_a1 = next_obs_a1
    obs_a2 = next_obs_a2
    obs_a3 = next_obs_a3
    
    # SAC updates
    if step > batch_size:
        a1_model.train(batch_size=batch_size, gradient_steps=1)
        a3_model.train(batch_size=batch_size, gradient_steps=1)
        
    # PPO update
    if a2_model.rollout_buffer.full:
        with torch.no_grad():
            next_obs_tensor_a2 = torch.tensor(obs_a2).unsqueeze(0).to("cpu")  # A2 is on CPU
            next_values_a2 = a2_model.policy.predict_values(next_obs_tensor_a2)[0]
        a2_model.rollout_buffer.compute_returns_and_advantage(next_values_a2, np.array([done]))
        a2_model.train()
        a2_model.rollout_buffer.reset()
        
    if done:
        obs_a1, _ = a1_env.reset()
        obs_a3, _ = a3_env.reset()
        obs_a2, _ = a2_env.reset()
        portfolio_returns = []
        prev_weights = np.zeros(50, dtype=np.float32)
        episode_start_a2 = True
        
    if step % 1000 == 0:
        print(f"Step {step}/{TOTAL_STEPS} | Latest Reward (Sharpe): {joint_reward:.4f}")
        
    if step % CKPT_INTERVAL == 0:
        tag = f"{step // 1000}k"
        a1_model.save(str(CKPT / f"a1_unified_joint_{tag}"))
        a2_model.save(str(CKPT / f"a2_unified_joint_{tag}"))
        a3_model.save(str(CKPT / f"a3_unified_joint_{tag}"))
        print(f"  [Checkpoint saved at step {step}]")

# ── 5. Save ───────────────────────────────────────────────────────────────────
print("\nSaving Jointly Fine-Tuned Models...")
a1_model.save(str(CKPT / "a1_unified_joint_final"))
a2_model.save(str(CKPT / "a2_unified_joint_final"))
a3_model.save(str(CKPT / "a3_unified_joint_final"))
print("Done.")
