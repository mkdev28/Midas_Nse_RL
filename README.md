# MIDAS-NSE: Multi-Modal India-Driven Adaptive Signal Architecture for NSE Portfolio Optimization

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-00a65a.svg)](https://github.com/DLR-RM/stable-baselines3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**MIDAS-NSE** is a production-grade, hierarchical Deep Reinforcement Learning (DRL) system engineered for dynamic, multi-modal asset allocation in the Indian stock market (NIFTY 50). 

While traditional algorithmic trading systems operate solely on price and technical indicators of individual stocks, MIDAS-NSE decouples macroeconomic capital allocation from individual equity selection. It introduces an India-specific, regime-aware architecture that coordinates three specialized RL agents over a shared latent market embedding, augmented by natural language processing (NLP) on financial news and explainable AI (XAI) attribution.

> **Status:** Engineering implementation, end-to-end inference pipelines, and latency optimizations are complete. Quantitative evaluation numbers and benchmark comparisons are being finalized for submission to an upcoming top-tier IEEE / ACM conference.

---

## Architectural Highlights

```text
[Multi-Modal Time-Series & Macro Signals] ──┐
  ├── NIFTY 50 OHLCV & Technicals            │
  ├── Macro (G-Sec 10Y Yield, Crude, Gold)   ├──► [Shared Unified Transformer Encoder] ──► Latent State (Z ∈ ℝ²⁵⁶)
  └── Institutional Flows (FII / DII Net)    │               (Frozen during RL)
                                             │
[Indian Financial News Headlines] ───────────┴──► [FinBERT Sentiment Module] (Macro F1 = 0.917)
                                                               │
                                                               ▼
        ┌──────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┐
        ▼                                                      ▼                                                      ▼
[Agent 1: Macro Allocator]                             [Agent 2: Sentiment Modifier]                          [Agent 3: Stock Picker]
  • Algorithm: SAC (Soft Actor-Critic)                   • Algorithm: PPO (Proximal Policy Opt.)                • Algorithm: SAC (Soft Actor-Critic)
  • Output: 4 Asset-Class Weights                        • Output: Asset Exposure Multipliers                   • Output: 50 NIFTY Equity Weights
  • Objective: Calmar Ratio (Crash Protection)           • Objective: Dynamic Return Delta                      • Objective: Sharpe Ratio - Correlation
        │                                                      │                                                      │
        └──────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                               ▼
                                                  [Multiplicative Coordinator]
                                          w_i = w_macro × m_sentiment × w_stock (Softmax)
                                                               │
                                                               ▼
                                               [Mechanistic XAI Layer (SHAP / Attn)]
                                                               │
                                                               ▼
                                             [Live Algorithmic Portfolio Execution]
```

### 1. Shared Unified Transformer Encoder
* **Multi-Modal State Compression:** Processes 60-day historical windows across 46 market signals—including NIFTY index technicals, macro commodities, currency exchange rates (INR/USD), RBI Repo rates, and India VIX regime classifications.
* **Pretraining & Freezing:** Pretrained on self-supervised and next-day return prediction objectives to generate a rich 256-dimensional representation ($Z \in \mathbb{R}^{256}$). The encoder weights are frozen during downstream reinforcement learning to prevent representation collapse in multi-agent environments.

### 2. Three Specialized Hierarchical DRL Agents
* **Macro Allocator (Agent 1 - SAC):** Allocates capital across four broad asset classes (Equities, Government Bonds, Commodities, Cash). Optimized for the **Calmar Ratio** to prioritize capital preservation and drawdown mitigation during severe market downturns.
* **Sentiment Modifier (Agent 2 - PPO):** Dynamically scales non-cash asset exposure using real-time news sentiment. Powered by a custom-fine-tuned **FinBERT** model trained on 27,000+ Indian financial news headlines, achieving an exceptional **91.7% Macro F1 score**.
* **Stock Picker (Agent 3 - SAC):** Selects individual portfolio weights across 50 NIFTY 50 equities. Optimized for risk-adjusted returns via a **Sharpe Ratio objective penalized by pairwise stock correlation** to guarantee portfolio diversification.

### 3. India-Specific Domain Engineering
* **DII Absorption Ratio:** Introduces a novel quantitative factor capturing how domestic institutional investors (DII) absorb liquidity during foreign institutional (FII) sell-offs—a critical microstructure dynamic unique to Indian equities.
* **Selective Replay Buffer:** Extends standard experience replay with permanent memory retention for tail-risk crisis regimes (e.g., historical flash crashes, election volatility, and VIX spikes $>25$), ensuring RL agents do not catastrophically forget defensive policies during extended bull markets.

### 4. Multiplicative Coordinator & Mechanistic XAI
* **Deterministic Aggregation:** Modulates macro allocation weights by news sentiment multipliers and individual stock selections: $w_{\text{final}, i} = \text{Softmax}(w_{\text{class}} \times m_{\text{sentiment}} \times w_{\text{stock}})$.
* **Mechanistic Interpretability:** Integrates **SHAP (SHapley Additive exPlanations)** gradient attribution and Transformer multi-head attention mapping. Our XAI layer verifies that the model systematically shifts attention to institutional FII/DII order flow imbalances during high-volatility crash regimes.

---

## Tech Stack

* **Core Development:** Python 3.12, PyTorch 2.0+, NumPy, Pandas, SciPy, Parquet
* **Deep Reinforcement Learning:** Stable-Baselines3 (`SAC`, `PPO`), OpenAI Gym / Farama Gymnasium
* **NLP & Sentiment Analysis:** Hugging Face Transformers (`ProsusAI/finbert`), PyTorch NLP
* **Explainable AI (XAI):** SHAP (`GradientExplainer`), Attention Heatmap Mapping
* **Performance Profiling:** Native PyTorch CPU evaluation profiling (`time.perf_counter`)
* **Visualization & Frontend:** Streamlit, Matplotlib, Seaborn, Plotly

---

## Repository Structure

```text
midas_nse/
├── app.py                     # Streamlit live interactive dashboard & portfolio explorer
├── data/
│   ├── raw/                   # Raw historical OHLCV, macro indicators, and institutional flows
│   ├── processed/             # Pre-computed multi-modal parquet/npy arrays & fitted scalers
│   └── external/              # FinBERT sentiment datasets & RBI/G-Sec macro yields
├── pipeline/
│   ├── features/              # Feature engineering, scaling, and DII absorption ratio math
│   ├── inference/             # End-to-end live EOD E2E inference & continuous learning loops
│   ├── eval/                  # Classical & literature baseline backtest evaluation engines
│   └── packaging/             # Custom OpenAI Gym / Gymnasium environment definitions
├── results/
│   ├── p10_validation/        # Validation sweeps, per-seed logs, and checkpoint manifests
│   ├── p11_baselines/         # Classical and literature baseline comparative evaluation outputs
│   └── p12_xai/               # SHAP attribution beeswarm plots & Transformer attention maps
└── checkpoints/               # Saved model artifacts (Transformer encoder, A1/A2/A3 SAC/PPO models)
```

---

## Setup & Installation

### Prerequisites
* Windows / Linux / macOS with Python 3.12+ installed.
* At least 8GB RAM (16GB recommended for multi-seed backtesting).

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mkdev28/midas-nse.git
   cd midas-nse
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install stable-baselines3 pandas numpy scipy pyarrow scikit-learn transformers shap streamlit matplotlib seaborn plotly
   ```

---

## Usage

### 1. Launch the Interactive Dashboard
To explore portfolio allocations, regime classifications, and live inference outputs visually:
```bash
streamlit run midas_nse/app.py
```

### 2. Run Deterministic Baseline Evaluations
To run evaluation scripts comparing the hierarchical RL agents against classical benchmarks (Equal Weight, Markowitz MVO, Static 60/40):
```bash
python midas_nse/pipeline/eval/p11_baselines.py
```

### 3. Generate XAI Attribution Reports
To run SHAP gradient explainability and extract attention heatmaps over tail-risk market events:
```bash
python midas_nse/pipeline/eval/p12_xai_global.py
```

### 4. Execute CPU Latency Profiling
To verify high-frequency intraday execution compliance under commodity CPU hardware:
```bash
python midas_nse/pipeline/eval/p13_latency_profiler.py
```
*(Note: Production end-to-end inference latency averages **~2.44 ms**, operating over two orders of magnitude below the NSE 500 ms algorithmic execution limit).*

---

## Quantitative Results & Benchmarks

To preserve blind review integrity and prevent self-scooping for an upcoming IEEE / ACM peer-reviewed conference submission, exact multi-seed quantitative performance metrics (Sharpe ratio, CAGR, maximum drawdown, and Calmar ratio comparisons against recent 2026 literature baselines) are currently withheld from this public README. 

Upon official publication, this section will be updated with comprehensive backtest tables, out-of-sample equity curves across the 2023–2025 test period, and ablation study breakdowns.

---

## Author & Contact

**Mohit Kankaria**
* **GitHub:** [@mkdev28](https://github.com/mkdev28)
* **Project Scope:** Deep Reinforcement Learning, Algorithmic Trading, Multi-Modal Time-Series Analysis, Explainable AI in Finance.

---
*Disclaimer: This repository is for academic research and algorithmic modeling demonstration purposes only. Nothing herein constitutes financial or investment advice.*
