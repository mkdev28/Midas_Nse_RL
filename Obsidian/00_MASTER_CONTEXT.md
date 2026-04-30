# MIDAS-NSE — Master Context

> **Purpose:** Paste this single file when starting a session with any LLM that doesn't have auto-file-loading (Claude, GPT, etc.). For Perplexity, use the Space — it loads files 01-04 automatically.
>
> **Last updated:** 2026-04-30

---

## Project Identity

**MIDAS-NSE** — Multi-modal India-Driven Adaptive Signal Architecture for NSE Portfolio Optimization

A Deep Reinforcement Learning portfolio manager for Indian NSE (NIFTY 50) stocks. Three specialized RL agents coordinate to decide daily portfolio weights across 50 stocks and cash.

**Target venue:** ACM ICAIF / IEEE TNNLS / NeurIPS FinML Workshop
**Hardware:** Intel i7 13620H, 16GB RAM, RTX 4060 8GB
**Python:** 3.12 | **OS:** Windows (PowerShell only)
**Project root:** `C:\Users\mohit\Projects and learning and practice\rl poject final\midas_nse\`

---

## Architecture (Locked — Do Not Change)

Three agents receive a shared 256-d embedding `Z` from a pretrained Transformer encoder:

- **Agent 1 (SAC):** Macro Allocator → 4 asset class weights (stocks/bonds/commodities/cash) → Calmar reward
- **Agent 2 (PPO):** Sentiment Modifier → 3 multipliers [0.5, 1.5] on non-cash classes → return delta reward
- **Agent 3 (SAC):** Stock Picker → 50 stock weights → Sharpe minus correlation penalty

Final weight: `class_weight × sentiment_modifier × stock_weight` → softmax normalized

All signals are **pre-computed to parquet/npy** before training. Zero API calls during `env.step()`.

---

## Current Status

| Phase | Status | Key Output |
|-------|--------|------------|
| D1-D9 Raw Data | ✅ Done | `data/raw/` — 9 datasets |
| P1 Calendar + Alignment | ✅ Done | `master_aligned.parquet` [4414, 25] |
| P2 Cleaning | ✅ Done | Validated, anomalies flagged |
| P3 Feature Engineering | ✅ Done | `features_v1.parquet` [4387, 45] |
| P4 Split + Normalize | ✅ Done | train/val/test.parquet + scaler.pkl |
| P5 Stock Features | ✅ Done | `stock_features.npy` [4414, 50, 12] |
| P6 Transformer Pretrain | ✅ Done (rerun needed after P8) | `transformer_encoder.pt` Z=256-d |
| **P7 FinBERT Sentiment** | **🔴 BLOCKED** | Not started |
| P8 Merge Sentiment | ⬜ Waiting on P7 | — |
| P6 Rerun | ⬜ Waiting on P8 | — |
| P9 Gym Environment | ⬜ Not started | — |
| P10 Agent Training | ⬜ Not started | — |
| P11 Backtesting | ⬜ Not started | — |
| P12 XAI | ⬜ Not started | — |

**Overall: ~55% complete. Current blocker is P7 (FinBERT sentiment).**

---

## Critical Warnings

1. **Sentiment is all zeros** in current parquets. Do NOT proceed to P9/P10 until P7+P8 complete and P6 is retrained.
2. **36 features** fed to Transformer must exactly match Gym environment at inference. Verify via `checkpoint['feature_cols']`.
3. **scaler.pkl** must be refit after P8 updates sentiment. Old scaler is invalid once sentiment is real.
4. **50 stocks, not 15.** Do not reduce. Agent 3 operates on all 50 × 12 features.
5. **No Reddit data.** Agent 2 uses only FinBERT sentiment + frozen A1 weights.
6. **Transformer is frozen** during RL training. Only agent policies update in P10.

---

## What To Do Now

**Read `04_NEXT_STEPS.md` for the exact task and commands.**

If that file is not available, the immediate next step is **P7: FinBERT Sentiment** — run the diagnostic on `kdave/Indian_Financial_News` dataset, then build the sentiment pipeline. See the full handoff for details.
