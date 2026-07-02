"""
p10_train_a3.py — Train Agent 3: SAC Stock Picker
- Obs: 856 = Z(256) + 50 stocks × 12 technical features
- Act: 50 stock weights -> softmax
- Reward: Sharpe - (0.1 × mean pairwise correlation)
- Timesteps: 500,000
- Device: CUDA (RTX 4060)
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pipeline.packaging.p9_gym_env import StockPickerEnv, MidasDataset, _MidasEncoder

PROC = Path("data/processed")
CKPT = Path("checkpoints")
LOGS = Path("logs/a3")
CKPT.mkdir(exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on : {device}")
if device == "cuda":
    print(f"GPU         : {torch.cuda.get_device_name(0)}")

ckpt = torch.load(CKPT / "transformer_encoder.pt", weights_only=False)
encoder = _MidasEncoder()
encoder.load_state_dict(ckpt["encoder_state"])
encoder.to(device)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
print("Encoder loaded — frozen ✅")

train_ds = MidasDataset(split="train")
val_ds = MidasDataset(split="val")
print(f"Train: {train_ds.T} steps | Val: {val_ds.T} steps")

def make_env(dataset):
    def _init():
        env = StockPickerEnv(dataset=dataset, encoder=encoder, device=device)
        return Monitor(env)
    return _init

train_env = DummyVecEnv([make_env(train_ds)])
eval_env = DummyVecEnv([make_env(val_ds)])

print(f"Obs dim : {train_env.observation_space.shape}")
print(f"Act dim : {train_env.action_space.shape}")

model = SAC(
    policy="MlpPolicy",
    env=train_env,
    device=device,
    learning_rate=3e-4,
    buffer_size=200_000,          # Fix 2: was 100k — doubled for longer episodes
    learning_starts=1_000,       # Fix 2: don't update before buffer has real data
    batch_size=512,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    target_entropy="auto",
    use_sde=False,
    policy_kwargs=dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
    tensorboard_log=str(LOGS),
    verbose=1,
)
print("SAC model created ✅")

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


checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path=str(CKPT),
    name_prefix="a3_sac",
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=str(CKPT / "a3_best"),  # Fix 1: isolated dir, won't clobber other checkpoints
    log_path=str(LOGS),
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    verbose=1,
)

class SharpeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and np.isfinite(info["episode"].get("r", np.nan)):
                self.episode_rewards.append(info["episode"]["r"])
        if len(self.episode_rewards) >= 10:
            vals = np.asarray(self.episode_rewards[-10:], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                self.logger.record("sharpe/mean_episode_reward", float(np.mean(vals)))
            self.episode_rewards = self.episode_rewards[-10:]
        # C5: log tail-risk buffer size
        if hasattr(self.model, 'replay_buffer') and hasattr(self.model.replay_buffer, 'tail_risk_count'):
            self.logger.record("c5/tail_risk_episodes",
                               self.model.replay_buffer.tail_risk_count)
        return True

sharpe_cb = SharpeLoggerCallback()

TOTAL_TIMESTEPS = 500_000
print(f"\nStarting A3 SAC — {TOTAL_TIMESTEPS:,} timesteps")
print("Checkpoints  -> checkpoints/a3_sac_*_steps.zip")
print("Best model   -> checkpoints/a3_best/best_model.zip")
print("TensorBoard  -> run: tensorboard --logdir logs/a3")
print("-" * 60)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, eval_cb, sharpe_cb],
    tb_log_name="a3_sac",
    reset_num_timesteps=True,
    progress_bar=True,
)

model.save(str(CKPT / "a3_sac_final"))
print("\nA3 training complete.")
print("   Final  -> checkpoints/a3_sac_final.zip")
print("   Best   -> checkpoints/a3_best/best_model.zip")
print("\nNext: run p10_validate_a1_a3.py for full checkpoint sweep.")