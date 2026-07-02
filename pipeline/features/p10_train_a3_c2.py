"""
p10_train_a3_c2.py — Train Agent 3 (Stock Picker) with C2 CrossModalTransformer encoder

Changes vs p10_train_a3.py (unified):
  1. Encoder : CrossModalTransformer (C2) instead of _MidasEncoder (unified)
  2. Paths   : checkpoints/c2/a3/  and  logs/a3_c2/  — NEVER touches checkpoints/a3_*
  3. Steps   : 150,000  (vs 500K unified — we know A3 overfits beyond 50K;
                         running to 150K gives us a broader sweep to confirm
                         or find a new peak with the richer C2 representations)
  4. Prefix  : a3_c2_sac (checkpoint filenames, TensorBoard tag)

Everything else (SAC hyperparams, C5 buffer, SharpeLoggerCallback) is
identical to the unified A3 script — no unnecessary divergence.

Obs: 856 = Z(256) + 50 stocks × 12 technical features
Act: 50 stock weights -> softmax
Reward: Sharpe - (0.1 × mean pairwise correlation)
Device: CUDA (RTX 4060)
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

# C2-specific import (replaces _MidasEncoder from unified script)
from models.cross_modal_transformer import CrossModalTransformer
from pipeline.packaging.p9_gym_env import StockPickerEnv, MidasDataset, SelectiveReplayBuffer

# ── Paths ─────────────────────────────────────────────────────────────────────
# CHANGE 2: All outputs go to c2/ subdirectories — never overwrites unified a3_*
PROC     = Path("data/processed")
CKPT     = Path("checkpoints")
CKPT_C2  = Path("checkpoints/c2/a3")        # C2 A3 checkpoints live here
LOGS     = Path("logs/a3_c2")               # C2 A3 TensorBoard lives here
CKPT_C2.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on : {device}")
if device == "cuda":
    print(f"GPU         : {torch.cuda.get_device_name(0)}")

# ── CHANGE 1: Load C2 Encoder (frozen) ───────────────────────────────────────
# W9:  checkpoint is a nested dict — use ckpt["encoder_state"]
# W18: keys are prefixed with "encoder." (saved from trainer wrapper) — must strip
ckpt = torch.load(CKPT / "c2" / "transformer_c2_encoder.pt", weights_only=False)
encoder = CrossModalTransformer()
raw_state = ckpt["encoder_state"]
stripped  = {k.replace("encoder.", "", 1): v for k, v in raw_state.items()}
encoder.load_state_dict(stripped)
encoder.to(device)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False          # W6: frozen during RL training
print(f"C2 Encoder loaded -- {len(ckpt['feature_cols'])} features, frozen [OK]")
print(f"Unified A3 baseline -> checkpoints/a3_sac_50000_steps.zip  (UNTOUCHED [OK])")

# ── Build Datasets ────────────────────────────────────────────────────────────
train_ds = MidasDataset(split="train")
val_ds   = MidasDataset(split="val")
print(f"Train: {train_ds.T} steps | Val: {val_ds.T} steps")

# ── Environments ──────────────────────────────────────────────────────────────
# Identical to unified A3 — only the encoder passed in has changed
def make_env(dataset):
    def _init():
        env = StockPickerEnv(dataset=dataset, encoder=encoder, device=device)
        return Monitor(env)
    return _init

train_env = DummyVecEnv([make_env(train_ds)])
eval_env  = DummyVecEnv([make_env(val_ds)])

print(f"Obs dim : {train_env.observation_space.shape}")   # (856,)
print(f"Act dim : {train_env.action_space.shape}")        # (50,)

# ── SAC Model ─────────────────────────────────────────────────────────────────
# Identical hyperparameters to unified A3 — deliberate, for a clean ablation
model = SAC(
    policy          = "MlpPolicy",
    env             = train_env,
    device          = device,
    learning_rate   = 3e-4,
    buffer_size     = 200_000,
    learning_starts = 1_000,
    batch_size      = 512,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    gradient_steps  = 1,
    ent_coef        = "auto",
    target_entropy  = "auto",
    use_sde         = False,
    policy_kwargs   = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU),
    tensorboard_log = str(LOGS),
    verbose         = 1,
)
print("SAC model created [OK]")

# ── C5: Replace SB3 default buffer with SelectiveReplayBuffer ─────────────────
# Identical to unified A3 — C5 is always on for all C2 agents
model.replay_buffer = SelectiveReplayBuffer(
    buffer_size          = 200_000,
    observation_space    = train_env.observation_space,
    action_space         = train_env.action_space,
    device               = device,
    n_envs               = 1,
    optimize_memory_usage= False,
)
print("C5: SelectiveReplayBuffer wired [OK]")
print(f"    VIX threshold  : {SelectiveReplayBuffer.VIX_THRESHOLD}")
print(f"    FII z-score    : {SelectiveReplayBuffer.FII_ZSCORE_THRESH}")
print(f"    Tail-risk ratio: {SelectiveReplayBuffer.TAIL_RISK_RATIO}")

# ── Callbacks ─────────────────────────────────────────────────────────────────
# CHANGE 4: name_prefix and save_path use c2 variants
checkpoint_cb = CheckpointCallback(
    save_freq   = 50_000,
    save_path   = str(CKPT_C2),
    name_prefix = "a3_c2_sac",     # -> a3_c2_sac_50000_steps.zip etc.
    verbose     = 1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = str(CKPT_C2 / "best"),
    log_path             = str(LOGS),
    eval_freq            = 10_000,
    n_eval_episodes      = 5,
    deterministic        = True,
    render               = False,
    verbose              = 1,
)

# Identical to unified A3 SharpeLoggerCallback — reused verbatim
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

# ── Train ─────────────────────────────────────────────────────────────────────
# CHANGE 3: 150K steps instead of 500K
# Rationale: Unified A3 showed overfitting beyond 50K (Sharpe 1.86→1.24 at 500K).
# Training to 150K gives us checkpoints at 50K, 100K, 150K to find the C2 peak.
# If C2 representations are richer, the peak may arrive later than 50K.
TOTAL_TIMESTEPS = 150_000
print(f"\nStarting A3_C2 SAC -- {TOTAL_TIMESTEPS:,} timesteps")
print("Checkpoints  -> checkpoints/c2/a3/a3_c2_sac_*_steps.zip  (every 50k)")
print("Best model   -> checkpoints/c2/a3/best/best_model.zip")
print("TensorBoard  -> run: tensorboard --logdir logs/a3_c2")
print("Baseline A3  -> checkpoints/a3_sac_50000_steps.zip  (UNTOUCHED [OK])")
print("-" * 60)

model.learn(
    total_timesteps     = TOTAL_TIMESTEPS,
    callback            = [checkpoint_cb, eval_cb, sharpe_cb],
    tb_log_name         = "a3_c2_sac",
    reset_num_timesteps = True,
    progress_bar        = True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(str(CKPT_C2 / "a3_c2_sac_final"))
print("\nA3_C2 training complete.")
print(f"   Checkpoints -> checkpoints/c2/a3/a3_c2_sac_50000_steps.zip")
print(f"               -> checkpoints/c2/a3/a3_c2_sac_100000_steps.zip")
print(f"               -> checkpoints/c2/a3/a3_c2_sac_150000_steps.zip")
print(f"   Final       -> checkpoints/c2/a3/a3_c2_sac_final.zip")
print(f"   Best (val)  -> checkpoints/c2/a3/best/best_model.zip")
print(f"\nBaseline A3 -> checkpoints/a3_sac_50000_steps.zip  (UNTOUCHED [OK])")
print("\nNext: eval with p10_eval_a3_c2.py and compare 50K/100K/150K vs unified 50K.")
