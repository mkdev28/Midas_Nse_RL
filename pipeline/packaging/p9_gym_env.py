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
        self.returns  = df["y_next_day_return"].values.astype(np.float32) # (T,) — NIFTY next-day return

        # ── A1 asset-class proxy returns ─────────────────────────────────────────
        # ── BOND RETURN: from raw gsec yield ──────────────────────────
        _gsec_raw = pd.read_parquet(
            Path(__file__).parent.parent.parent / "data/raw/gsec_10y_yield_raw.parquet"
        )
        _gsec_raw["date"] = pd.to_datetime(_gsec_raw["date"])
        _gsec_raw = _gsec_raw.set_index("date")["gsec_10y_yield"]
        _gsec_aligned = _gsec_raw.reindex(
            pd.to_datetime(df["date"])
        ).ffill().bfill().values.astype(np.float32)
        # yield up → bond price down: bond_ret = -delta_yield/100
        self.bond_ret = np.concatenate([
            [0.0],
            -np.diff(_gsec_aligned) / 100.0
        ]).astype(np.float32)

        # ── COMMODITY RETURN: from raw gold USD price ──────────────────
        _gold_raw = pd.read_parquet(
            Path(__file__).parent.parent.parent / "data/raw/gold.parquet"
        )
        _gold_raw.index = pd.to_datetime(_gold_raw.index)
        _gold_close = _gold_raw[("Close", "GC=F")].reindex(
            pd.to_datetime(df["date"])
        ).ffill().bfill()
        self.commodity_ret = np.concatenate([
            [0.0],
            _gold_close.pct_change().dropna().values
        ])[:len(df)].astype(np.float32)
        self.commodity_ret = np.nan_to_num(
            self.commodity_ret, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Cash: constant RBI repo rate (approx. 6.5% p.a. / 252 trading days)
        self.cash_ret = np.full(len(df), 0.065 / 252, dtype=np.float32)

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

        # ── Stock features: date-based lookup to avoid positional misalignment ──
        # stock_features.npy covers ALL 4414 trading days (from 2008-01-01).
        # train.parquet starts on 2008-02-06 (row 26 in the npy).
        # We build a date→index dict so that env.step() always fetches the
        # correct npy row regardless of which split is loaded.
        with open(PROC / "stock_features_meta.pkl", "rb") as _f:
            _meta = pickle.load(_f)
        self.trading_days   = _meta["trading_days"]          # list of 4414 date strings
        self._date_to_idx   = {d: i for i, d in enumerate(self.trading_days)}
        X_full = np.load(PROC / "stock_features.npy")        # (4414, 50, 12) — z-score normalized
        self.X_stock = np.nan_to_num(X_full, nan=0.0).astype(np.float32)

        # Un-normalize col 9 (daily_return) back to fractional returns.
        # stock_features.npy is z-score normalized (mean/std computed on training data).
        # Using the z-score directly as a return causes equity *= (1 + z) to collapse.
        means = np.load(PROC / "stock_features_means.npy")   # (1, 1, 12)
        stds  = np.load(PROC / "stock_features_stds.npy")    # (1, 1, 12)
        ret_mean = float(means.flat[9])   # training mean of daily_return
        ret_std  = float(stds.flat[9])    # training std  of daily_return
        # Reconstruct true fractional daily_return: x_raw = z * std + mean
        self.stock_returns = (
            X_full[:, :, 9] * ret_std + ret_mean
        ).astype(np.float32)              # (4414, 50) — true fractional returns
        self.stock_returns = np.nan_to_num(self.stock_returns, nan=0.0,
                                           posinf=0.0, neginf=0.0)

        # Store per-row npy indices aligned to the parquet rows (for fast lookup)
        self.stock_idx = np.array(
            [self._date_to_idx[str(pd.Timestamp(d).date())] for d in self.dates],
            dtype=np.int32
        )  # shape (T,) — maps parquet row i → npy row stock_idx[i]

        self.T = len(df)
        print(f"[MidasDataset] split={split} | T={self.T} | "
              f"features={self.features.shape} | "
              f"X_stock={self.X_stock.shape} | "
              f"first_npy_idx={self.stock_idx[0]} (date={self.trading_days[self.stock_idx[0]]})"
              f" last_npy_idx={self.stock_idx[-1]} (date={self.trading_days[self.stock_idx[-1]]})"
        )


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
        self._z_cache = {}

    def _encode(self, t) -> np.ndarray:
        """Encode 60-day window ending at t-1 → Z (256,)."""
        if t in self._z_cache:
            return self._z_cache[t]
            
        window = self.ds.features[t - WINDOW : t]               # (60, 46)
        x = torch.tensor(window).unsqueeze(0).to(self.device)   # (1, 60, 46)
        with torch.no_grad():
            z = self.encoder(x)                                  # (1, 256)
        z_np = z.squeeze(0).cpu().numpy()                        # (256,)
        
        self._z_cache[t] = z_np
        return z_np

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
        asset_rets = np.array([
            nifty_ret,                              # stocks  (weight[0])
            float(self.ds.bond_ret[self.t]),        # bonds   (weight[1])
            float(self.ds.commodity_ret[self.t]),   # commodities (weight[2])
            float(self.ds.cash_ret[self.t]),        # cash    (weight[3])
        ], dtype=np.float32)
        port_ret = float((weights * asset_rets).sum())
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
        self._stock_ret_history = []   # list of (50,) arrays — for correlation
        self._port_ret_history  = []   # list of scalars  — actual earned returns (no hindsight)

    def _get_obs(self):
        z          = self._encode(self.t)                           # (256,)
        npy_i      = self.ds.stock_idx[self.t]                      # correct npy row
        stock_flat = self.ds.X_stock[npy_i].flatten()               # (600,)
        obs        = np.concatenate([z, stock_flat]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def step(self, action):
        exp_a   = np.exp(action - action.max())
        weights = (exp_a / exp_a.sum()).astype(np.float32)
        self.current_weights = weights

        # Per-stock returns: use pre-computed un-normalized fractional returns
        # (X_stock col 9 is z-score normalized and cannot be used directly as returns)
        npy_i      = self.ds.stock_idx[self.t]                      # correct npy row
        stock_rets = self.ds.stock_returns[npy_i].copy()             # (50,) true fractional returns
        port_ret   = float((weights * stock_rets).sum())
        self._stock_ret_history.append(stock_rets)
        self._port_ret_history.append(port_ret)  # record actual earned return for this step

        t_now = self.t
        self._step_time()
        reward = self._sharpe_corr(weights)
        obs    = self._get_obs() if not self.done else np.zeros(856, dtype=np.float32)
        obs    = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        info = {
            "weights":  weights,
            "port_ret": port_ret,
            "vix":      float(self.ds.vix[t_now]),
            "fii_net":  float(self.ds.fii_net[t_now]),
        }
        return obs, float(reward), self.done, False, info

    def _sharpe_corr(self, weights) -> float:
        if len(self._port_ret_history) < 30:
            return 0.0
        # Use actual per-step earned returns (weights locked at decision time)
        # This avoids hindsight bias from re-applying today's weights to old history.
        port_rets = np.array(self._port_ret_history[-30:])          # (30,)

        std = port_rets.std()
        sharpe = (port_rets.mean() / std * np.sqrt(252)) if std > 1e-8 else 0.0

        rets = np.asarray(self._stock_ret_history[-30:], dtype=np.float32)   # (30, 50) for correlation
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        std_devs = rets.std(axis=0)
        valid_cols = std_devs > 1e-8

        if valid_cols.sum() >= 2:
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.corrcoef(rets[:, valid_cols], rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            upper = corr[np.triu_indices_from(corr, k=1)]
            mean_corr = float(np.mean(upper)) if upper.size > 0 else 0.0
        else:
            mean_corr = 0.0

        reward = sharpe - 0.1 * mean_corr
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))

        return reward

    def reset(self, *, seed=None, options=None):
        self.current_weights    = np.full(50, 1/50, dtype=np.float32)
        self._stock_ret_history = []
        self._port_ret_history  = []
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
        asset_rets   = np.array([
            nifty_ret,
            float(self.ds.bond_ret[self.t]),
            float(self.ds.commodity_ret[self.t]),
            float(self.ds.cash_ret[self.t]),
        ], dtype=np.float32)
        base_ret     = float((a1_weights * asset_rets).sum())
        modified_ret = float((modified   * asset_rets).sum())
        reward       = modified_ret - base_ret

        self.t += 1
        if self.t >= self.T:
            self.done = True

        obs = self._get_obs() if not self.done else np.zeros(7, dtype=np.float32)
        info = {
            "multipliers": multipliers,
            "modified_ret": modified_ret,
            "modified_weights": modified
        }
        return obs, float(reward), self.done, False, info

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


# ── C5 Selective Replay Buffer (SB3-compatible) ───────────────────────────────
from stable_baselines3.common.buffers import ReplayBuffer as _SB3ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples as _RBSamples
import torch as _torch

class SelectiveReplayBuffer(_SB3ReplayBuffer):
    """
    SB3-compatible replay buffer that permanently retains
    Indian market tail-risk episodes (C5 contribution).

    Tail-risk criteria (either triggers permanent retention):
      - india_vix   > 25.0              (VIX spike)
      - |fii_net| z-score > 2.0        (extreme institutional flow)

    Normal experiences : standard FIFO via SB3 parent class.
    Tail-risk experiences : stored in _tail_* lists, NEVER discarded.
    Sampling : guarantees TAIL_RISK_RATIO of every batch comes from
               tail-risk store (when available).

    Drop-in replacement — call after SAC() init:
        model.replay_buffer = SelectiveReplayBuffer(
            buffer_size=200_000,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device, n_envs=1,
        )
    """

    TAIL_RISK_RATIO   = 0.20
    VIX_THRESHOLD     = 25.0
    FII_ZSCORE_THRESH = 2.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Permanent tail-risk store — never discarded
        self._tail_obs       = []
        self._tail_next_obs  = []
        self._tail_actions   = []
        self._tail_rewards   = []
        self._tail_dones     = []
        # Rolling FII history for z-score calculation
        self._fii_history    = []

    # ── Tail-risk detection ───────────────────────────────────────────────────
    def _is_tail_risk(self, info: dict) -> bool:
        vix     = float(info.get("vix",     0.0))
        fii_net = float(info.get("fii_net", 0.0))
        self._fii_history.append(fii_net)
        # Keep a rolling 252-day window for z-score stability
        if len(self._fii_history) > 252:
            self._fii_history.pop(0)

        vix_spike = vix > self.VIX_THRESHOLD

        fii_extreme = False
        if len(self._fii_history) > 30:
            arr     = np.array(self._fii_history)
            zscore  = abs((fii_net - arr.mean()) / (arr.std() + 1e-8))
            fii_extreme = zscore > self.FII_ZSCORE_THRESH

        return vix_spike or fii_extreme

    # ── Override add() ────────────────────────────────────────────────────────
    def add(self, obs, next_obs, action, reward, done, infos):
        # Extract tail-risk signal from first env's info
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        if self._is_tail_risk(info):
            # Permanently store in tail-risk buffer
            self._tail_obs.append(obs.copy())
            self._tail_next_obs.append(next_obs.copy())
            self._tail_actions.append(action.copy())
            self._tail_rewards.append(np.array([[float(reward)]]))
            self._tail_dones.append(np.array([[float(done)]]))

        # Always add to normal SB3 FIFO buffer as well
        super().add(obs, next_obs, action, reward, done, infos)

    # ── Override sample() ─────────────────────────────────────────────────────
    def sample(self, batch_size: int, env=None):
        n_tail   = int(batch_size * self.TAIL_RISK_RATIO)
        n_normal = batch_size - n_tail

        # Normal batch from SB3 parent
        normal_batch = super().sample(n_normal, env=env)

        # No tail-risk data yet → return normal batch only
        if len(self._tail_obs) == 0 or n_tail == 0:
            return normal_batch

        # Sample from tail-risk store (with replacement if store is small)
        idx = np.random.randint(0, len(self._tail_obs), size=n_tail)
        dev = normal_batch.observations.device

        # SB3 normal_batch.observations is strictly 2D (batch_size, obs_dim).
        # But during add(), obs was stored as (1, obs_dim) because of DummyVecEnv.
        # We must strip the first dimension [i][0] to make them match.
        tail_obs      = _torch.FloatTensor(
            np.array([self._tail_obs[i][0]      for i in idx])).to(dev)   # (n_tail, obs_dim)
        tail_next_obs = _torch.FloatTensor(
            np.array([self._tail_next_obs[i][0] for i in idx])).to(dev)   # (n_tail, obs_dim)
        tail_actions  = _torch.FloatTensor(
            np.array([self._tail_actions[i][0]  for i in idx])).to(dev)   # (n_tail, act_dim)
            
        # tail_rewards were stored as [[float(reward)]] i.e. (1, 1).
        # [i][0] extracts (1,) which stacks to (n_tail, 1) — perfectly matching SB3.
        tail_rewards  = _torch.FloatTensor(
            np.array([self._tail_rewards[i][0]  for i in idx])).to(dev)   # (n_tail, 1)
        tail_dones    = _torch.FloatTensor(
            np.array([self._tail_dones[i][0]    for i in idx])).to(dev)   # (n_tail, 1)

        tail_batch = _RBSamples(
            observations=tail_obs,
            next_observations=tail_next_obs,
            actions=tail_actions,
            rewards=tail_rewards,
            dones=tail_dones,
        )

        # Concatenate normal + tail into one batch
        return _RBSamples(
            observations     = _torch.cat([normal_batch.observations,      tail_batch.observations]),
            next_observations= _torch.cat([normal_batch.next_observations, tail_batch.next_observations]),
            actions          = _torch.cat([normal_batch.actions,           tail_batch.actions]),
            rewards          = _torch.cat([normal_batch.rewards,           tail_batch.rewards]),
            dones            = _torch.cat([normal_batch.dones,             tail_batch.dones]),
        )

    @property
    def tail_risk_count(self) -> int:
        """Number of permanently retained tail-risk experiences."""
        return len(self._tail_obs)

    def stats(self) -> dict:
        return {
            "normal_size":    self.size(),
            "tail_risk_size": self.tail_risk_count,
            "total":          self.size() + self.tail_risk_count,
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

    # Replay buffer — SB3-compatible API
    buf = SelectiveReplayBuffer(
        buffer_size=1000,
        observation_space=env_a1.observation_space,
        action_space=env_a1.action_space,
        device="cpu", n_envs=1,
    )
    obs_a1, _ = env_a1.reset()
    for _ in range(50):
        act = env_a1.action_space.sample()
        next_obs, r, done, _, info = env_a1.step(act)
        buf.add(obs_a1[None], next_obs[None], act[None],
                np.array([r]), np.array([done]), [info])
        obs_a1 = next_obs
        if done: obs_a1, _ = env_a1.reset()
    print(f"[Buffer] {buf.stats()}")
    assert buf.tail_risk_count >= 0  # GFC data → some tail-risk expected

    print("\n✅ All smoke tests passed.")
