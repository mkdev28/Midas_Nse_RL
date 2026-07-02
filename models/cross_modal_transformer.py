"""
models/cross_modal_transformer.py
C2 Cross-Modal Transformer for MIDAS-NSE.

Architecture:
  Input: x (batch, window=60, 46) — identical format to unified Transformer

  Step 1 — Split by modality:
    x_macro [B, 60, 23]  → MacroEncoder     → tok_macro [B, 60, 128]
    x_tech  [B, 60, 16]  → TechnicalEncoder → tok_tech  [B, 60, 128]
    x_sent  [B, 60, 7]   → SentimentEncoder → tok_sent  [B, 60, 128]

  Step 2 — Add modality type embeddings (learnable):
    Each modality gets its own type vector so Transformer distinguishes sources.
    tok_macro += emb(MODALITY_MACRO)   i.e. embedding[0] broadcast over T
    tok_tech  += emb(MODALITY_TECH)    embedding[1]
    tok_sent  += emb(MODALITY_SENT)    embedding[2]

  Step 3 — Add sinusoidal positional encoding (shared):
    Same position index for same timestep across modalities.
    Temporal ordering is preserved; cross-modal comes from type embeddings.

  Step 4 — Concatenate: [B, 3×60=180, 128]

  Step 5 — TransformerEncoder (d=128, nhead=4, layers=3):
    Cross-modal attention: macro tokens can attend to sentiment tokens, etc.

  Step 6 — Mean pool over 180 tokens → [B, 128]
           Linear(128 → 256) → Z [B, 256]

Output Z(256) matches unified Transformer output dim.
No agent retraining scripts need editing — drop-in for _MidasEncoder.

Checkpoint isolation:
  Unified:   checkpoints/transformer_encoder.pt    (UNTOUCHED)
  C2:        checkpoints/c2/transformer_c2_encoder.pt
  Full C2:   checkpoints/c2/transformer_c2_best.pt
"""

import math
import torch
import torch.nn as nn
from pathlib import Path
from models.encoders import (
    MacroEncoder, TechnicalEncoder, SentimentEncoder,
    MACRO_IDX, TECHNICAL_IDX, SENTIMENT_IDX,
)

# Modality IDs — match the Embedding(3, d_enc) indices
MODALITY_MACRO     = 0
MODALITY_TECHNICAL = 1
MODALITY_SENTIMENT = 2

# Checkpoint path (C2 outputs go here — never touches unified checkpoints)
C2_CKPT = Path("checkpoints/c2")


class CrossModalTransformer(nn.Module):
    """
    C2 Cross-Modal Transformer.
    Drop-in replacement for _MidasEncoder — same input format (B, 60, 46),
    same output dim Z (B, 256). Use encode() for agent observations.
    """

    def __init__(self,
                 d_enc:    int   = 128,    # encoder output dim (intermediate)
                 d_out:    int   = 256,    # final Z dim — must match unified
                 nhead:    int   = 4,
                 n_layers: int   = 3,
                 window:   int   = 60,
                 dropout:  float = 0.1):
        super().__init__()
        self.d_enc  = d_enc
        self.d_out  = d_out
        self.window = window

        # ── Modality-specific encoders ────────────────────────────────────────
        self.macro_enc = MacroEncoder(d_enc)
        self.tech_enc  = TechnicalEncoder(d_enc)
        self.sent_enc  = SentimentEncoder(d_enc)

        # ── Modality type embeddings ──────────────────────────────────────────
        # 3 learnable vectors: tells Transformer which modality each token is.
        # This is the key C2 innovation vs unified Transformer (which has none).
        self.modality_emb = nn.Embedding(3, d_enc)

        # ── Sinusoidal positional encoding ────────────────────────────────────
        # Shared across modalities — same position = same timestep.
        self.register_buffer("pos_enc",
                             self._build_pos_enc(window, d_enc))

        # ── Cross-modal TransformerEncoder ────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_enc, nhead=nhead,
            dim_feedforward=d_enc * 4,    # 512 for d_enc=128
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=n_layers)
        self.norm = nn.LayerNorm(d_enc)

        # ── Output projection ─────────────────────────────────────────────────
        # Mean pool 180 tokens → d_enc → project to d_out (256)
        # Keeps Z dim identical to unified Transformer for agent compatibility.
        self.out_proj = nn.Linear(d_enc, d_out)

    @staticmethod
    def _build_pos_enc(max_len: int, d_model: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, window, 46)  — same input as unified Transformer
        Returns: Z (batch, 256)
        """
        B, T, _ = x.shape
        dev = x.device

        # ── Step 1: Split by modality ─────────────────────────────────────────
        x_macro = x[:, :, MACRO_IDX]      # (B, T, 23)
        x_tech  = x[:, :, TECHNICAL_IDX]  # (B, T, 16)
        x_sent  = x[:, :, SENTIMENT_IDX]  # (B, T, 7)

        # ── Step 2: Encode each modality ──────────────────────────────────────
        tok_macro = self.macro_enc(x_macro)  # (B, T, 128)
        tok_tech  = self.tech_enc(x_tech)    # (B, T, 128)
        tok_sent  = self.sent_enc(x_sent)    # (B, T, 128)

        # ── Step 3: Add modality type embeddings ──────────────────────────────
        ids_macro = torch.full((B, T), MODALITY_MACRO,     dtype=torch.long, device=dev)
        ids_tech  = torch.full((B, T), MODALITY_TECHNICAL, dtype=torch.long, device=dev)
        ids_sent  = torch.full((B, T), MODALITY_SENTIMENT, dtype=torch.long, device=dev)

        tok_macro = tok_macro + self.modality_emb(ids_macro)
        tok_tech  = tok_tech  + self.modality_emb(ids_tech)
        tok_sent  = tok_sent  + self.modality_emb(ids_sent)

        # ── Step 4: Add positional encoding (same position = same timestep) ───
        pe = self.pos_enc[:, :T, :].to(dev)  # (1, T, d_enc)
        tok_macro = tok_macro + pe
        tok_tech  = tok_tech  + pe
        tok_sent  = tok_sent  + pe

        # ── Step 5: Concatenate → (B, 3T, d_enc) ─────────────────────────────
        # Token layout: [0:T]=macro, [T:2T]=technical, [2T:3T]=sentiment
        tokens = torch.cat([tok_macro, tok_tech, tok_sent], dim=1)

        # ── Step 6: Cross-modal attention ─────────────────────────────────────
        out = self.transformer(tokens)   # (B, 3T, d_enc)
        out = self.norm(out)

        # ── Step 7: Mean pool + project ───────────────────────────────────────
        z = out.mean(dim=1)              # (B, d_enc=128)
        z = self.out_proj(z)             # (B, d_out=256)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias matching _MidasEncoder.encode() interface. Used by gym envs."""
        return self.forward(x)

    def get_attention_weights(self, x: torch.Tensor):
        """
        Extract per-layer attention weights for XAI (C3).

        Returns: list of (B, nhead, 3T, 3T) tensors, one per Transformer layer.
        Token layout for interpretation:
          rows/cols [0:T]    = macro tokens     (price, VIX, FII/DII)
          rows/cols [T:2T]   = technical tokens (RSI, MACD, BB, ATR, C1 India)
          rows/cols [2T:3T]  = sentiment tokens (FinBERT daily score + stats)

        Off-diagonal blocks show cross-modal attention:
          rows [0:T]   × cols [2T:3T] = macro→sentiment attention
          rows [T:2T]  × cols [0:T]   = technical→macro attention  etc.
        """
        weights = []

        def _hook(module, inp, out):
            # Re-run self-attention to get weights (need_weights=True)
            q = k = v = inp[0]
            _, attn = module.self_attn(q, k, v,
                                        need_weights=True,
                                        average_attn_weights=False)
            weights.append(attn.detach().cpu())

        handles = [layer.register_forward_hook(_hook)
                   for layer in self.transformer.layers]
        with torch.no_grad():
            self.forward(x)
        for h in handles:
            h.remove()

        return weights   # list of (B, nhead, 3T, 3T), one per layer


# ── Pretraining wrapper (same task as unified: predict next-day return) ────────
class CrossModalTransformerPretrainer(nn.Module):
    """
    Wraps CrossModalTransformer with a regression head for pretraining.
    Pretraining task: predict y_next_day_return (same as unified Transformer).
    Input : (batch, window, 46)
    Output: (batch,) — scalar next-day return prediction
    """

    def __init__(self, window: int = 60, d_out: int = 256):
        super().__init__()
        self.encoder = CrossModalTransformer(window=window, d_out=d_out)
        self.head    = nn.Linear(d_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)             # (B, 256)
        return self.head(z).squeeze(-1) # (B,)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Used by gym envs — returns Z (B, 256)."""
        return self.encoder(x)
