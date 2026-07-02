"""
p10_train_a1.py — Train Agent 1: SAC Macro Allocator
- Obs: 265 = Z(256) + 5 macro + 4 current weights
- Act: 4 class weights (stocks, bonds, commodities, cash) -> softmax
- Reward: Calmar ratio (annualized return / max drawdown, 60-day rolling)
- Timesteps: 500,000
- Device: CUDA (RTX 4060)
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pipeline.packaging.p9_gym_env import MacroAllocatorEnv, MidasDataset, _MidasEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC = Path("data/processed")
CKPT = Path("checkpoints")
LOGS = Path("logs/a1_unified_fixed")
CKPT.mkdir(exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on : {device}")
if device == "cuda":
    print(f"GPU         : {torch.cuda.get_device_name(0)}")

# ── Load Encoder (frozen) ─────────────────────────────────────────────────────
# W9: encoder checkpoint is a nested dict — load via ckpt["encoder_state"]
# W10: input dim is 46, not 36
ckpt = torch.load(CKPT / "transformer_encoder.pt", weights_only=False)
encoder = _MidasEncoder()
encoder.load_state_dict(ckpt["encoder_state"])
encoder.to(device)
encoder.eval()
for param in encoder.parameters():
    param.requires_grad = False          # W6: frozen during RL training
print(f"Encoder loaded — {len(ckpt['feature_cols'])} features, frozen ✅")

# ── Build Datasets ────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = MidasDataset(split="train")
val_ds   = MidasDataset(split="val")
print(f"Train: {train_ds.T} steps | Val: {val_ds.T} steps")

# ── Environments ──────────────────────────────────────────────────────────────
def make_env(dataset):
    def _init():
        env = MacroAllocatorEnv(dataset=dataset, encoder=encoder, device=device)
        return Monitor(env)
    return _init

train_env = DummyVecEnv([make_env(train_ds)])
eval_env  = DummyVecEnv([make_env(val_ds)])

print(f"Obs dim : {train_env.observation_space.shape}")    # (265,)
print(f"Act dim : {train_env.action_space.shape}")         # (4,)

# ── SAC Model ─────────────────────────────────────────────────────────────────
model = SAC(
    policy           = "MlpPolicy",
    env              = train_env,
    device           = device,
    learning_rate    = 3e-4,
    buffer_size      = 200_000,          # Fix 2: was 100k — doubled for longer episodes
    learning_starts  = 1_000,           # Fix 3: don't update before buffer has real data
    batch_size       = 512,
    tau              = 0.005,
    gamma            = 0.99,
    train_freq       = 1,
    gradient_steps   = 1,
    ent_coef         = "auto",
    target_entropy   = "auto",
    use_sde          = False,
    policy_kwargs    = dict(
        net_arch     = [256, 256],
        activation_fn = torch.nn.ReLU,
    ),
    tensorboard_log  = str(LOGS),
    verbose          = 1,
)
print(f"SAC model created ✅")

# ── C5: Replace SB3 default buffer with SelectiveReplayBuffer ─────────────────
from pipeline.packaging.p9_gym_env import SelectiveReplayBuffer

model.replay_buffer = SelectiveReplayBuffer(
    buffer_size          = 200_000,
    observation_space    = train_env.observation_space,
    action_space         = train_env.action_space,
    device               = device,
    n_envs               = 1,
    optimize_memory_usage= False,
)
print("C5: SelectiveReplayBuffer wired ✅")
print(f"    VIX threshold  : {SelectiveReplayBuffer.VIX_THRESHOLD}")
print(f"    FII z-score    : {SelectiveReplayBuffer.FII_ZSCORE_THRESH}")
print(f"    Tail-risk ratio: {SelectiveReplayBuffer.TAIL_RISK_RATIO}")


# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq    = 50_000,
    save_path    = str(CKPT),
    name_prefix  = "a1_unified_fixed",
    verbose      = 1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = str(CKPT / "a1_unified_fixed_best"),   # Fix 1: isolated dir, won't overwrite A3 best_model.zip
    log_path             = str(LOGS),
    eval_freq            = 10_000,
    n_eval_episodes      = 5,
    deterministic        = True,
    render               = False,
    verbose              = 1,
)

class CalmarLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        if len(self.episode_rewards) >= 10:
            self.logger.record("calmar/mean_episode_reward",
                               np.mean(self.episode_rewards[-10:]))
            self.episode_rewards = self.episode_rewards[-10:]
        # C5: log tail-risk buffer size
        if hasattr(self.model, 'replay_buffer') and hasattr(self.model.replay_buffer, 'tail_risk_count'):
            self.logger.record("c5/tail_risk_episodes",
                               self.model.replay_buffer.tail_risk_count)
        return True

calmar_cb = CalmarLoggerCallback()

# ── Train ─────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000
print(f"\nStarting A1 SAC — {TOTAL_TIMESTEPS:,} timesteps")
print("Checkpoints  → checkpoints/a1_unified_fixed_*_steps.zip  (every 50k steps)")
print("Best model   → checkpoints/a1_unified_fixed_best/best_model.zip")
print("TensorBoard  → run: tensorboard --logdir logs/a1_unified_fixed")
print("-" * 60)

model.learn(
    total_timesteps     = TOTAL_TIMESTEPS,
    callback            = [checkpoint_cb, eval_cb, calmar_cb],
    tb_log_name         = "a1_unified_fixed",
    reset_num_timesteps = True,
    progress_bar        = True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(str(CKPT / "a1_unified_fixed_final"))
print("\nA1 training complete.")
print(f"   Final  -> checkpoints/a1_unified_fixed_final.zip")
print(f"   Best   -> checkpoints/a1_unified_fixed_best/best_model.zip")
print("\nNext: run p10_validate_a1_a3.py to compare checkpoint sweep.")