# pipeline/packaging/p9_gym_env.py
"""
MIDAS-NSE — P9 Gymnasium Environment
Three gym envs for A1 (SAC Macro), A2 (PPO Sentiment), A3 (SAC Stock).
All data pre-loaded at init. env.step() = array lookups only.
"""

import numpy as np
import pandas as pd
import torch
import pickle
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import math
import torch.nn as nn
# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW       = 60
D_MODEL      = 256
N_STOCKS     = 50
N_STOCK_FEAT = 12
PROC         = Path("data/processed")
CKPT         = Path("checkpoints")


# ── Encoder Loader ────────────────────────────────────────────────────────────


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class _MidasEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(46, 256)
        self.pos_enc    = _PositionalEncoding(max_len=500)
        enc_layer       = nn.TransformerEncoderLayer(d_model=256, nhead=4,
                              dim_feedforward=512, dropout=0.1, batch_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.norm       = nn.LayerNorm(256)
    def encode(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.norm(x)[:, -1, :]
    def forward(self, x):
        return self.encode(x)

def load_encoder(device="cpu"):
    ckpt = torch.load(CKPT / "transformer_encoder.pt",
                      map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        model = _MidasEncoder()
        state = ckpt["encoder_state"] if isinstance(ckpt, dict) and "encoder_state" in ckpt else ckpt; model.load_state_dict(state)
    else:
        model = ckpt
    model.to(device)
    model.eval()
    return model


# ── Dataset ───────────────────────────────────────────────────────────────────
class MidasDataset:
    """
    Pre-loads everything at init. No I/O during training.
    """
    def __init__(self, split="train"):
        assert split in ("train", "val", "test")

        df = pd.read_parquet(PROC / f"{split}.parquet").reset_index()
        df["date"] = pd.to_datetime(df["date"])
        self.dates = df["date"].values

        # Feature columns for transformer (46 features, excludes target)
        ckpt = torch.load(CKPT / "transformer_best.pt", map_location="cpu", weights_only=False)
        self.feature_cols = ckpt["feature_cols"]
        self.features = df[self.feature_cols].values.astype(np.float32)   # (T, 46)
        self.returns  = df["y_next_day_return"].values.astype(np.float32) # (T,)

        # A1 macro signals: VIX, G-Sec yield, INR/USD, FII net, daily sentiment
        macro_cols = ["vix_close", "gsec_10y_yield", "inrusd_close",
                      "fii_net", "daily_score"]
        self.macro = df[macro_cols].values.astype(np.float32)             # (T, 5)

        # A2 sentiment signals
        sent_cols  = ["daily_score", "sentiment_5dma", "sentiment_momentum"]
        self.sentiment = df[sent_cols].values.astype(np.float32)          # (T, 3)

        # VIX and FII for replay buffer tail-risk tagging
        self.vix     = df["vix_close"].values.astype(np.float32)          # (T,)
        self.fii_net = df["fii_net"].values.astype(np.float32)         # (T,)

        # Stock features — fix 26-row warmup offset on train split only
        offset = 26 if split == "train" else 0
        X = np.load(PROC / f"X_{split}_technical.npy")                   # (T+offset, 50, 12)
        X = X[offset:]
        self.stock_features = np.nan_to_num(X, nan=0.0).astype(np.float32)  # (T, 50, 12)

        self.T = len(df)
        assert self.T == len(self.stock_features), (
            f"[MidasDataset] ALIGNMENT ERROR: parquet={self.T} "
            f"npy={len(self.stock_features)}"
        )
        print(f"[MidasDataset] split={split} | T={self.T} | "
              f"features={self.features.shape} | stocks={self.stock_features.shape}")


# ── Base Environment ──────────────────────────────────────────────────────────
class MidasBaseEnv(gym.Env):
    """Shared window slicing, encoder call, and time stepping."""

    def __init__(self, dataset: MidasDataset, encoder, device="cpu"):
        super().__init__()
        self.ds      = dataset
        self.encoder = encoder
        self.device  = device
        self.T       = dataset.T
        self.t       = WINDOW
        self.done    = False

    def _encode(self, t) -> np.ndarray:
        """Encode 60-day window ending at t-1 → Z (256,)."""
        window = self.ds.features[t - WINDOW : t]               # (60, 46)
        x = torch.tensor(window).unsqueeze(0).to(self.device)   # (1, 60, 46)
        with torch.no_grad():
            z = self.encoder(x)                                  # (1, 256)
        return z.squeeze(0).cpu().numpy()                        # (256,)

    def _step_time(self):
        self.t += 1
        if self.t >= self.T:
            self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t    = WINDOW
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


# ── A1 — SAC Macro Allocator ──────────────────────────────────────────────────
class MacroAllocatorEnv(MidasBaseEnv):
    """
    State  : Z(256) + 5 macro + 4 current weights = 265
    Action : 4 raw logits → softmax → [stocks, bonds, commodities, cash]
    Reward : Calmar ratio — annualized return / max drawdown (60-day rolling)
    """

    def __init__(self, dataset, encoder, device="cpu"):
        super().__init__(dataset, encoder, device)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(265,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.current_weights  = np.full(4, 0.25, dtype=np.float32)
        self._return_history  = []

    def _get_obs(self):
        z     = self._encode(self.t)        # (256,)
        macro = self.ds.macro[self.t]       # (5,)
        return np.concatenate([z, macro, self.current_weights]).astype(np.float32)

    def step(self, action):
        # Softmax normalize → class weights
        exp_a = np.exp(action - action.max())
        weights = (exp_a / exp_a.sum()).astype(np.float32)
        self.current_weights = weights

        nifty_ret = float(self.ds.returns[self.t])
        port_ret  = weights[0] * nifty_ret  # stock class drives return
        self._return_history.append(port_ret)

        t_now = self.t
        self._step_time()
        reward = self._calmar()
        obs    = self._get_obs() if not self.done else np.zeros(265, dtype=np.float32)

        info = {
            "weights":   weights,
            "port_ret":  port_ret,
            "vix":       float(self.ds.vix[t_now]),
            "fii_net":   float(self.ds.fii_net[t_now]),
        }
        return obs, float(reward), self.done, False, info

    def _calmar(self) -> float:
        if len(self._return_history) < WINDOW:
            return 0.0
        rets    = np.array(self._return_history[-WINDOW:])
        ann_ret = rets.mean() * 252
        cum     = np.cumprod(1 + rets)
        dd      = cum / np.maximum.accumulate(cum) - 1
        max_dd  = abs(dd.min())
        return ann_ret / max_dd if max_dd > 1e-8 else ann_ret

    def reset(self, *, seed=None, options=None):
        self.current_weights = np.full(4, 0.25, dtype=np.float32)
        self._return_history = []
        return super().reset(seed=seed, options=options)


# ── A3 — SAC Stock Picker ─────────────────────────────────────────────────────
class StockPickerEnv(MidasBaseEnv):
    """
    State  : Z(256) + 50 × 12 = 856
    Action : 50 raw logits → softmax → stock weights
    Reward : Sharpe − 0.1 × mean pairwise correlation (30-day rolling)
    """

    def __init__(self, dataset, encoder, device="cpu"):
        super().__init__(dataset, encoder, device)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(856,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(50,), dtype=np.float32
        )
        self.current_weights    = np.full(50, 1/50, dtype=np.float32)
        self._stock_ret_history = []   # list of (50,) arrays

    def _get_obs(self):
        z          = self._encode(self.t)                           # (256,)
        stock_flat = self.ds.stock_features[self.t].flatten()       # (600,)
        return np.concatenate([z, stock_flat]).astype(np.float32)

    def step(self, action):
        exp_a   = np.exp(action - action.max())
        weights = (exp_a / exp_a.sum()).astype(np.float32)
        self.current_weights = weights

        # Per-stock returns: stock_features col 9 = daily_return
        stock_rets = self.ds.stock_features[self.t, :, 9].copy()   # (50,)
        port_ret   = float((weights * stock_rets).sum())
        self._stock_ret_history.append(stock_rets)

        t_now = self.t
        self._step_time()
        reward = self._sharpe_corr(weights)
        obs    = self._get_obs() if not self.done else np.zeros(856, dtype=np.float32)

        info = {
            "weights":  weights,
            "port_ret": port_ret,
            "vix":      float(self.ds.vix[t_now]),
            "fii_net":  float(self.ds.fii_net[t_now]),
        }
        return obs, float(reward), self.done, False, info

    def _sharpe_corr(self, weights) -> float:
        if len(self._stock_ret_history) < 30:
            return 0.0
        hist      = np.array(self._stock_ret_history[-30:])  # (30, 50)
        port_rets = (hist * weights).sum(axis=1)              # (30,)

        std = port_rets.std()
        sharpe = (port_rets.mean() / std * np.sqrt(252)) if std > 1e-8 else 0.0

        corr   = np.corrcoef(hist.T)                          # (50, 50)
        mask   = np.triu(np.ones((50, 50), dtype=bool), k=1)
        mean_corr = corr[mask].mean()

        return sharpe - 0.1 * mean_corr

    def reset(self, *, seed=None, options=None):
        self.current_weights    = np.full(50, 1/50, dtype=np.float32)
        self._stock_ret_history = []
        return super().reset(seed=seed, options=options)


# ── A2 — PPO Sentiment Modifier ───────────────────────────────────────────────
class SentimentModifierEnv(gym.Env):
    """
    State  : 3 sentiment signals + 4 frozen A1 weights = 7
    Action : 3 multipliers ∈ [0.5, 1.5] for non-cash classes
    Reward : Return delta vs A1 baseline
    Note   : No transformer call. Built only after A1 is frozen.
    """

    def __init__(self, dataset: MidasDataset, a1_weights_fn):
        """
        a1_weights_fn : callable(t) → np.ndarray(4,)
            Query your frozen A1 model: lambda t: a1_model.predict(obs_at_t)[0]
        """
        super().__init__()
        self.ds            = dataset
        self.a1_weights_fn = a1_weights_fn
        self.T             = dataset.T
        self.t             = WINDOW
        self.done          = False

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.5, high=1.5, shape=(3,), dtype=np.float32
        )

    def _get_obs(self):
        sent       = self.ds.sentiment[self.t]     # (3,)
        a1_weights = self.a1_weights_fn(self.t)    # (4,)
        return np.concatenate([sent, a1_weights]).astype(np.float32)

    def step(self, action):
        multipliers = np.clip(action, 0.5, 1.5).astype(np.float32)
        a1_weights  = self.a1_weights_fn(self.t)   # (4,)

        # Apply multipliers to non-cash classes, renormalize
        modified       = a1_weights.copy()
        modified[:3]  *= multipliers
        modified       = modified / modified.sum()

        nifty_ret    = float(self.ds.returns[self.t])
        base_ret     = a1_weights[0] * nifty_ret
        modified_ret = modified[0]   * nifty_ret
        reward       = modified_ret - base_ret

        self.t += 1
        if self.t >= self.T:
            self.done = True

        obs = self._get_obs() if not self.done else np.zeros(7, dtype=np.float32)
        return obs, float(reward), self.done, False, {"multipliers": multipliers}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t    = WINDOW
        self.done = False
        return self._get_obs(), {}


# ── Coordinator ───────────────────────────────────────────────────────────────
def coordinate(
    class_weights: np.ndarray,          # (4,) from A1 — softmax
    sentiment_multipliers: np.ndarray,  # (3,) from A2 — [0.5, 1.5]
    stock_weights: np.ndarray,          # (50,) from A3 — softmax
) -> np.ndarray:
    """
    Deterministic math coordinator. No neural net.
    Returns: final_stock_weights (50,) summing to 1.
    """
    modified       = class_weights.copy()
    modified[:3]  *= sentiment_multipliers
    modified       = modified / modified.sum()          # renormalize

    final = modified[0] * stock_weights                 # stock class × per-stock weight
    return (final / final.sum()).astype(np.float32)     # normalize to sum=1


# ── C5 Selective Replay Buffer ────────────────────────────────────────────────
class SelectiveReplayBuffer:
    """
    C5 implementation.
    Normal   : FIFO, capacity 100,000
    Tail-risk: unlimited, never discarded
    Batch    : 80% normal / 20% tail-risk
    Criteria : VIX > 25 OR |FII net z-score| > 2
    """

    NORMAL_CAPACITY = 100_000
    TAIL_RATIO      = 0.20

    def __init__(self):
        self.normal     = []
        self.tail_risk  = []
        self._fii_hist  = []

    def _is_tail_risk(self, vix: float, fii_net: float) -> bool:
        self._fii_hist.append(fii_net)
        if len(self._fii_hist) > 252:
            self._fii_hist.pop(0)
        if len(self._fii_hist) > 10:
            mu    = np.mean(self._fii_hist)
            sigma = np.std(self._fii_hist) + 1e-8
            fii_z = abs((fii_net - mu) / sigma)
        else:
            fii_z = 0.0
        return vix > 25.0 or fii_z > 2.0

    def add(self, transition: dict, vix: float, fii_net: float):
        if self._is_tail_risk(vix, fii_net):
            self.tail_risk.append(transition)
        else:
            if len(self.normal) >= self.NORMAL_CAPACITY:
                self.normal.pop(0)
            self.normal.append(transition)

    def sample(self, batch_size: int) -> list:
        n_tail   = max(1, int(batch_size * self.TAIL_RATIO))
        n_normal = batch_size - n_tail

        batch = []
        if self.tail_risk:
            idx   = np.random.choice(len(self.tail_risk),
                                     min(n_tail, len(self.tail_risk)), replace=False)
            batch += [self.tail_risk[i] for i in idx]
        if self.normal:
            idx   = np.random.choice(len(self.normal),
                                     min(n_normal, len(self.normal)), replace=False)
            batch += [self.normal[i] for i in idx]
        return batch

    def __len__(self):
        return len(self.normal) + len(self.tail_risk)

    def stats(self) -> dict:
        return {
            "normal_size":    len(self.normal),
            "tail_risk_size": len(self.tail_risk),
            "total":          len(self),
        }


# ── Factory ───────────────────────────────────────────────────────────────────
def make_envs(split="train", device="cpu"):
    """
    Call this to get A1 and A3 envs.
    A2 requires a trained A1 first — use make_a2_env() after A1 is frozen.
    """
    print(f"[P9] Loading {split} dataset...")
    dataset = MidasDataset(split=split)

    print(f"[P9] Loading transformer encoder...")
    encoder = load_encoder(device=device)

    env_a1 = MacroAllocatorEnv(dataset, encoder, device)
    env_a3 = StockPickerEnv(dataset,   encoder, device)

    print(f"[P9] Ready.")
    print(f"     A1: obs={env_a1.observation_space.shape}  act={env_a1.action_space.shape}")
    print(f"     A3: obs={env_a3.observation_space.shape}  act={env_a3.action_space.shape}")
    return env_a1, env_a3, dataset, encoder


def make_a2_env(split="train", a1_weights_fn=None):
    """
    Build A2 env only after A1 is trained and frozen.
    a1_weights_fn: callable(t) → np.ndarray(4,)
    """
    assert a1_weights_fn is not None, \
        "Pass a1_weights_fn — e.g. lambda t: a1_model.predict(obs_at_t)[0]"
    dataset = MidasDataset(split=split)
    env_a2  = SentimentModifierEnv(dataset, a1_weights_fn)
    print(f"[P9] A2: obs={env_a2.observation_space.shape}  act={env_a2.action_space.shape}")
    return env_a2


# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MIDAS-NSE P9 — Smoke Test")
    print("=" * 60)

    env_a1, env_a3, dataset, encoder = make_envs(split="train")

    # A1
    obs, _ = env_a1.reset()
    assert obs.shape == (265,), f"A1 obs wrong: {obs.shape}"
    obs, rew, done, _, info = env_a1.step(env_a1.action_space.sample())
    print(f"\n[A1] reward={rew:.6f} | weights={info['weights'].round(3)}")

    # A3
    obs, _ = env_a3.reset()
    assert obs.shape == (856,), f"A3 obs wrong: {obs.shape}"
    obs, rew, done, _, info = env_a3.step(env_a3.action_space.sample())
    print(f"[A3] reward={rew:.6f} | max_stock_wt={info['weights'].max():.4f}")

    # A2 with dummy A1
    dummy_a1 = lambda t: np.array([0.4, 0.2, 0.2, 0.2], dtype=np.float32)
    env_a2   = make_a2_env(split="train", a1_weights_fn=dummy_a1)
    obs, _   = env_a2.reset()
    assert obs.shape == (7,), f"A2 obs wrong: {obs.shape}"
    obs, rew, done, _, info = env_a2.step(env_a2.action_space.sample())
    print(f"[A2] reward={rew:.6f} | multipliers={info['multipliers'].round(3)}")

    # Coordinator
    cw  = np.array([0.4, 0.2, 0.25, 0.15])
    sm  = np.array([1.1, 0.9, 1.2])
    sw  = np.ones(50) / 50
    fw  = coordinate(cw, sm, sw)
    assert abs(fw.sum() - 1.0) < 1e-6, "Coordinator weights don't sum to 1"
    print(f"\n[Coordinator] final_weights sum={fw.sum():.6f} ✅")

    # Replay buffer
    buf = SelectiveReplayBuffer()
    buf.add({"s": 1, "a": 0}, vix=32.0, fii_net=-800)  # tail-risk
    buf.add({"s": 2, "a": 1}, vix=15.0, fii_net=200)   # normal
    print(f"[Buffer] {buf.stats()}")
    assert buf.stats()["tail_risk_size"] == 1
    assert buf.stats()["normal_size"]    == 1

    print("\n✅ All smoke tests passed.")
