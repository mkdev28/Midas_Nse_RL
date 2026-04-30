# MIDAS-NSE — Decisions Log

> **Purpose:** Records *why* each major architectural and design decision was made. Prevents future LLMs from re-debating settled questions.
> **Last updated:** 2026-04-30

---

## How to Use

When a decision is made during a session, add it here with:
- The decision (one line)
- The alternatives considered
- Why this option was chosen
- Date and which session/LLM made it

---

## Architecture Decisions

### D1 — Three-agent hierarchy instead of single agent
**Decision:** Use Agent 1 (macro) → Agent 2 (sentiment modifier) → Agent 3 (stock picker) instead of a single flat RL agent.
**Alternatives:** Single SAC/PPO agent with all features concatenated; two-agent (macro + stock); four-agent (adding a risk agent).
**Rationale:** Three agents map to the three distinct decision layers in real portfolio management. Also enables C1 (hierarchical DRL) and C3 (per-agent XAI). Single agent with 856+ features would be extremely hard to train and interpret.
**Date:** Project inception

### D2 — SAC for Agents 1 and 3, PPO for Agent 2
**Decision:** SAC for the two core allocation agents, PPO for the modifier.
**Alternatives:** All SAC, all PPO, TD3, DDPG.
**Rationale:** SAC handles continuous action spaces well and has entropy regularization (important for exploration in portfolio weights). PPO for Agent 2 because it's a simpler 7-d state → 3-d action problem and PPO is more stable for small action spaces.
**Date:** Project inception

### D3 — Transformer encoder pretrained via MSE on next-day return
**Decision:** Pretrain the shared encoder to predict `y_next_day_return` via MSELoss, then freeze and use the last-timestep embedding as Z.
**Alternatives:** Autoencoder reconstruction, contrastive learning, no pretraining (train end-to-end with RL).
**Rationale:** Return prediction forces the encoder to learn market-relevant representations. Freezing during RL prevents catastrophic forgetting and reduces compute.
**Date:** Pre-P6

### D4 — RobustScaler instead of StandardScaler
**Decision:** Use RobustScaler for normalization.
**Alternatives:** StandardScaler, MinMaxScaler, no scaling.
**Rationale:** Financial data has heavy tails and outliers (2008 crash, 2020 COVID). RobustScaler uses median/IQR instead of mean/std, so outliers don't distort the scaling.
**Date:** P4

### D5 — 60-day lookback window for Transformer
**Decision:** Use 60 trading days (~3 calendar months) as the input window.
**Alternatives:** 20 days (1 month), 120 days (6 months), 252 days (1 year).
**Rationale:** 60 days captures medium-term trends without excessive memory usage. Balances between too-short (noisy) and too-long (computationally expensive, dilutes recent information).
**Date:** Pre-P6

### D6 — DII Absorption Ratio as key India signal
**Decision:** Use `dii_net / abs(fii_net)` on FII-negative days as a feature.
**Alternatives:** Raw FII/DII flows, FII-DII spread, binary regime indicator.
**Rationale:** This ratio specifically captures the structural regime shift in Indian markets (2022-2024) where DIIs began absorbing FII selling. It's the core of research contribution C1.
**Date:** Project design

### D7 — Selective replay buffer for tail-risk events
**Decision:** Permanently retain experiences where VIX > 25 or FII net z-score > 2. Normal experiences use FIFO.
**Alternatives:** Uniform replay, prioritized experience replay (PER), no special handling.
**Rationale:** Tail-risk events are rare (~5% of trading days) but critical. Standard replay forgets them as they get pushed out by normal experiences. PER upweights high-TD-error samples but doesn't guarantee crisis retention.
**Date:** Research contribution C5 design

### D8 — Crude oil negative price retained
**Decision:** Keep the April 2020 negative crude oil price as real data.
**Alternatives:** Clip to zero, drop the day, interpolate.
**Rationale:** It actually happened. The model should learn from extreme events, not have them sanitized away. An `extreme_event` flag was added instead.
**Date:** D5 data download

---

*Add new decisions below this line.*
