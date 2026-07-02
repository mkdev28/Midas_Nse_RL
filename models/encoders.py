"""
models/encoders.py
Modality-specific encoders for MIDAS-NSE C2 contribution.

Feature splits verified against transformer_best.pt FEATURE_COLS (46 features):
  MACRO_IDX      : [0:23]  — price (4) + macro (8) + FII/DII (11) = 23 features
  TECHNICAL_IDX  : [23:39] — technical indicators (13) + C1 India signals (3) = 16 features
  SENTIMENT_IDX  : [39:46] — FinBERT sentiment signals = 7 features

Column order (from p6_transformer_pretrain.py FEATURE_COLS):
  [0]  nifty_close     [1]  nifty_high      [2]  nifty_low       [3]  nifty_open
  [4]  vix_close       [5]  gold_close      [6]  crude_close     [7]  inrusd_close
  [8]  repo_rate       [9]  gsec_10y_yield  [10] yield_spread    [11] gold_crude_ratio
  [12] fii_net         [13] dii_net         [14] fii_buy_value   [15] fii_sell_value
  [16] dii_buy_value   [17] dii_sell_value  [18] fii_net_5d      [19] fii_net_20d
  [20] dii_net_5d      [21] dii_net_20d     [22] fii_dii_net_5d
  ── MACRO boundary at 23 ──
  [23] ret_1d          [24] ret_5d          [25] ret_20d         [26] rsi_14
  [27] macd            [28] macd_signal     [29] macd_hist       [30] bb_upper
  [31] bb_lower        [32] bb_width        [33] bb_pct          [34] atr_14
  [35] atr_pct
  [36] dii_absorption_ratio   [37] vix_regime   [38] institutional_net   ← C1 India signals
  ── TECHNICAL boundary at 39 ──
  [39] daily_score     [40] daily_pos_count [41] daily_neg_count
  [42] sentiment_5dma  [43] sentiment_vol   [44] sentiment_momentum
  [45] sentiment_available
  ── SENTIMENT boundary at 46 ──
"""

import torch
import torch.nn as nn

# ── Column index slices (do NOT change — verified against checkpoint) ──────────
MACRO_IDX     = slice(0,  23)   # 23 features: price + macro + FII/DII
TECHNICAL_IDX = slice(23, 39)   # 16 features: technical indicators + C1 India
SENTIMENT_IDX = slice(39, 46)   #  7 features: FinBERT sentiment signals


class MacroEncoder(nn.Module):
    """
    MLP encoder for price + macro + FII/DII signals.
    Using MLP (not CNN) because macro signals are scalar time-series where
    positional ordering within the window matters less than temporal patterns
    captured by the Transformer.

    Input : (batch, window, 23)
    Output: (batch, window, d_model)
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 23)
        return self.norm(self.net(x))                      # (B, T, d_model)


class TechnicalEncoder(nn.Module):
    """
    1D CNN encoder for technical indicators + C1 India signals.
    CNN is preferred over MLP here because technical indicators have strong
    local temporal structure (momentum, RSI crossovers, MACD divergence).
    Conv1d captures these patterns across the time axis.

    Input : (batch, window, 16)
    Output: (batch, window, d_model)
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 16)
        x = x.permute(0, 2, 1)   # (B, 16, T) for Conv1d
        x = self.conv(x)          # (B, d_model, T)
        x = x.permute(0, 2, 1)   # (B, T, d_model)
        return self.norm(x)


class SentimentEncoder(nn.Module):
    """
    Linear projection encoder for FinBERT sentiment signals.
    Sentiment signals are already compressed (7-d aggregates from 768-d FinBERT
    embeddings), so a simple linear projection is sufficient. No CNN needed —
    sentiment changes are episodic, not locally structured.

    Input : (batch, window, 7)
    Output: (batch, window, d_model)
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, d_model),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 7)
        return self.norm(self.net(x))                      # (B, T, d_model)
