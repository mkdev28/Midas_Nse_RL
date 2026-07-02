import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pipeline.packaging.p9_gym_env import SentimentModifierEnv, MacroAllocatorEnv, MidasDataset, _MidasEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
CKPT = Path("checkpoints")
LOGS = Path("logs/a2_unified")
LOGS.mkdir(parents=True, exist_ok=True)

device = "cpu"
print(f"Training A2 (Unified) on : {device}")

# ── Load Unified Encoder ──────────────────────────────────────────────────────
ckpt = torch.load(CKPT / "transformer_encoder.pt", map_location=device, weights_only=False)
encoder = _MidasEncoder()
encoder.load_state_dict(ckpt["encoder_state"])
encoder.to(device)
encoder.eval()

# ── Load Unified A1 Model ─────────────────────────────────────────────────────
a1_model_path = CKPT / "a1_unified_fixed_300000_steps.zip"
print(f"Loading A1 model from: {a1_model_path}")
a1_model = SAC.load(a1_model_path, device=device)

# ── Precompute A1 Weights ─────────────────────────────────────────────────────
def precompute_a1_weights(split):
    ds = MidasDataset(split=split)
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

print("Precomputing A1 unified weights for Train split...")
train_a1_weights, train_ds = precompute_a1_weights("train")
print("Precomputing A1 unified weights for Val split...")
val_a1_weights, val_ds = precompute_a1_weights("val")

# ── Build Environments ────────────────────────────────────────────────────────
def make_env(dataset, weights):
    def _init():
        env = SentimentModifierEnv(dataset=dataset, a1_weights_fn=lambda t: weights[t - 60])
        return Monitor(env)
    return _init

train_env = DummyVecEnv([make_env(train_ds, train_a1_weights)])
eval_env  = DummyVecEnv([make_env(val_ds, val_a1_weights)])

print(f"Obs dim : {train_env.observation_space.shape}")
print(f"Act dim : {train_env.action_space.shape}")

# ── PPO Model ─────────────────────────────────────────────────────────────────
model = PPO(
    policy           = "MlpPolicy",
    env              = train_env,
    device           = device,
    learning_rate    = 3e-4,
    n_steps          = 2048,
    batch_size       = 64,
    n_epochs         = 10,
    gamma            = 0.99,
    gae_lambda       = 0.95,
    clip_range       = 0.2,
    ent_coef         = 0.0,
    tensorboard_log  = str(LOGS),
    verbose          = 1,
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq    = 50_000,
    save_path    = str(CKPT),
    name_prefix  = "a2_unified",
    verbose      = 1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = str(CKPT / "a2_unified_best"),
    log_path             = str(LOGS),
    eval_freq            = 10_000,
    n_eval_episodes      = 5,
    deterministic        = True,
    render               = False,
    verbose              = 1,
)

# ── Train ─────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 300_000
print(f"\nStarting A2 PPO (Unified) — {TOTAL_TIMESTEPS:,} timesteps")
print("-" * 60)

model.learn(
    total_timesteps     = TOTAL_TIMESTEPS,
    callback            = [checkpoint_cb, eval_cb],
    tb_log_name         = "a2_unified",
    reset_num_timesteps = True,
    progress_bar        = True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(str(CKPT / "a2_unified_final"))
print("\nA2 Unified training complete.")
