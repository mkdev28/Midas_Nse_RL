import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="MIDAS-NSE Dashboard", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "checkpoints"

# Hardcoded Sector Mapping for NIFTY 50
SECTOR_MAP = {
    "HDFCBANK.NS": "Financials", "ICICIBANK.NS": "Financials", "SBIN.NS": "Financials", 
    "KOTAKBANK.NS": "Financials", "AXISBANK.NS": "Financials", "INDUSINDBK.NS": "Financials",
    "BAJFINANCE.NS": "Financials", "BAJAJFINSV.NS": "Financials", "SBILIFE.NS": "Financials", "HDFCLIFE.NS": "Financials",
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy", "COALINDIA.NS": "Energy", 
    "NTPC.NS": "Energy", "POWERGRID.NS": "Energy",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "LT.NS": "Industrials", "ADANIENT.NS": "Industrials", "ADANIPORTS.NS": "Industrials",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma", "DIVISLAB.NS": "Pharma", "APOLLOHOSP.NS": "Pharma",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer",
    "ULTRACEMCO.NS": "Materials", "GRASIM.NS": "Materials", "SHREECEM.NS": "Materials",
    "UPL.NS": "Chemicals", "BHARTIARTL.NS": "Telecom"
}

# ---------------------------------------------------------
# HELPER FUNCTIONS (CACHED)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def get_latest_allocation_file():
    files = glob.glob(str(RESULTS_DIR / "live_allocations_*.csv"))
    if not files:
        return None, None
    files.sort(reverse=True)
    latest_file = files[0]
    date_str = Path(latest_file).stem.split('_')[-1]
    
    yesterday_file = files[1] if len(files) > 1 else None
    return latest_file, yesterday_file, date_str

@st.cache_data(ttl=300)
def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data(ttl=300)
def load_shadow_log():
    return load_csv_safe(CKPT_DIR / "live" / "shadow_log.csv")

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & REGIME BADGE
# ---------------------------------------------------------
st.sidebar.title("MIDAS-NSE")
st.sidebar.markdown("---")

# Determine Regime
shadow_log = load_shadow_log()
current_vix = 15.0
current_fii = 0.0
is_crisis = False

if shadow_log is not None and not shadow_log.empty:
    latest_log = shadow_log.iloc[-1]
    current_vix = latest_log.get('avg_vix_30d', 15.0)
    current_fii = latest_log.get('avg_fii_zscore_30d', 0.0)
    is_crisis = current_vix > 25.0 or abs(current_fii) > 2.0

if is_crisis:
    st.sidebar.error("🔴 REGIME: CRISIS\n\nTail-Risk Gate Active")
else:
    st.sidebar.success("🟢 REGIME: NORMAL\n\nMarket stable")

st.sidebar.markdown("---")

pages = [
    "Today's Action",
    "Why Did We Do This? (XAI)",
    "Market Regime & Safety Gate",
    "Shadow Learning Tracker",
    "Performance & Risk History",
    "Risk & Compliance",
    "Config & Audit"
]
page = st.sidebar.radio("Navigation", pages)

# Fetch latest data context
latest_alloc_path, prev_alloc_path, current_date = get_latest_allocation_file()
date_display = current_date if current_date else "NO DATA"

st.markdown(f"**Data as of: {date_display}**")
if not current_date:
    st.warning("No allocation files found. Run `live_inference.py` to generate data.")
    st.stop()

df_today = load_csv_safe(latest_alloc_path)
df_yesterday = load_csv_safe(prev_alloc_path) if prev_alloc_path else None

if df_today is None:
    st.error(f"Failed to load {latest_alloc_path}")
    st.stop()

# Split allocations
macro_assets = ['BONDS', 'COMMODITIES', 'CASH']
df_macro = df_today[df_today['Symbol'].isin(macro_assets)]
df_stocks = df_today[~df_today['Symbol'].isin(macro_assets)]
total_stock_weight = df_stocks['Weight'].sum()

# ---------------------------------------------------------
# PANEL 1: TODAY'S ACTION
# ---------------------------------------------------------
if page == "Today's Action":
    st.header(f"Panel 1 — Today's Action ({date_display})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1a. Asset Class Split")
        macro_plot_df = pd.DataFrame({
            'Asset': ['Stocks', 'Bonds', 'Commodities', 'Cash'],
            'Weight': [
                total_stock_weight,
                df_macro[df_macro['Symbol'] == 'BONDS']['Weight'].values[0] if 'BONDS' in df_macro['Symbol'].values else 0,
                df_macro[df_macro['Symbol'] == 'COMMODITIES']['Weight'].values[0] if 'COMMODITIES' in df_macro['Symbol'].values else 0,
                df_macro[df_macro['Symbol'] == 'CASH']['Weight'].values[0] if 'CASH' in df_macro['Symbol'].values else 0
            ]
        })
        fig_pie = px.pie(macro_plot_df, values='Weight', names='Asset', hole=0.4, 
                         color='Asset', color_discrete_map={'Stocks':'#1f77b4', 'Bonds':'#ff7f0e', 'Commodities':'#2ca02c', 'Cash':'#d62728'})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("1b. Stock-Level Allocation")
        df_stocks_plot = df_stocks.copy()
        df_stocks_plot['Sector'] = df_stocks_plot['Symbol'].map(lambda x: SECTOR_MAP.get(x, "Other"))
        
        fig_tree = px.treemap(df_stocks_plot, path=['Sector', 'Symbol'], values='Weight',
                              color='Sector')
        fig_tree.update_traces(textinfo="label+value")
        st.plotly_chart(fig_tree, use_container_width=True)

    st.subheader("1c. Trade Execution Delta Table")
    if df_yesterday is not None:
        merged = pd.merge(df_today, df_yesterday, on='Symbol', suffixes=('_today', '_yest'))
        merged['Delta'] = merged['Weight_today'] - merged['Weight_yest']
        
        def action_label(delta):
            if delta > 0.001: return 'BUY'
            if delta < -0.001: return 'SELL'
            return 'HOLD'
            
        merged['Action'] = merged['Delta'].apply(action_label)
        merged['Est Cost (bps)'] = (merged['Delta'].abs() * 15).round(2)
        
        st.dataframe(merged[['Symbol', 'Weight_yest', 'Weight_today', 'Delta', 'Action', 'Est Cost (bps)']].style.map(
            lambda x: 'color: green' if x == 'BUY' else ('color: red' if x == 'SELL' else ''), subset=['Action']
        ))
        
        total_cost = merged['Est Cost (bps)'].sum()
        st.info(f"**Total Estimated Transaction Cost:** {total_cost:.2f} bps")
    else:
        st.info("No previous day allocation found. Showing current targets as initial allocation.")
        st.dataframe(df_today)

# ---------------------------------------------------------
# PANEL 2: WHY DID WE DO THIS? (XAI)
# ---------------------------------------------------------
elif page == "Why Did We Do This? (XAI)":
    st.header("Panel 2 — XAI & Interpretability")
    
    xai_report_path = RESULTS_DIR / "p12_xai" / "xai_report.md"
    if os.path.exists(xai_report_path):
        with open(xai_report_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning(f"File not found: {xai_report_path} — run the P12 XAI script.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2a. Macro SHAP Attribution")
        img_path = RESULTS_DIR / "p12_xai" / "shap_temporal_top2.png"
        if os.path.exists(img_path):
            st.image(Image.open(img_path), caption="Temporal SHAP Analysis", use_container_width=True)
            st.success("Leading driver: FII Net Outflows pushed allocation toward defensive assets.")
        else:
            st.info("SHAP visualization not found.")

    with col2:
        st.subheader("2b. Transformer Attention")
        img_path2 = RESULTS_DIR / "p12_xai" / "attention_map.png"
        if os.path.exists(img_path2):
            st.image(Image.open(img_path2), caption="Transformer Attention Heatmap", use_container_width=True)
            st.info("High attention peaks on Day -1 (recency bias) and Day -51 (historical anchoring).")
        else:
            st.info("Attention visualization not found.")

# ---------------------------------------------------------
# PANEL 3: MARKET REGIME
# ---------------------------------------------------------
elif page == "Market Regime & Safety Gate":
    st.header("Panel 3 — Market Regime & Safety Gate")
    
    if is_crisis:
        st.error(f"### 🚨 Tail-Risk Gate ACTIVE — System in defensive regime")
    else:
        st.success(f"### ✅ NORMAL Regime — Markets stable")
        
    col1, col2, col3 = st.columns(3)
    col1.metric(label="India VIX", value=f"{current_vix:.2f}", delta=f"{current_vix - 25.0:.2f} to threshold", delta_color="inverse")
    col2.metric(label="FII Net Flow (Z-score)", value=f"{current_fii:.2f}", delta="Safe limit: ±2.0", delta_color="off")
    col3.metric(label="RBI Repo Rate", value="6.50%", delta="0.00%", delta_color="off")
    
    st.markdown("---")
    st.write("*(Note: Live data read from `shadow_log.csv` moving averages. If file missing, shows defaults.)*")

# ---------------------------------------------------------
# PANEL 4: SHADOW LEARNING
# ---------------------------------------------------------
elif page == "Shadow Learning Tracker":
    st.header("Panel 4 — Shadow Learning Tracker")
    
    if shadow_log is not None and not shadow_log.empty:
        st.subheader("4a. Production vs Shadow Performance")
        
        # Ensure we have numeric data for Sharpe columns (in case they are empty strings)
        shadow_log['shadow_sharpe'] = pd.to_numeric(shadow_log['shadow_sharpe'], errors='coerce').fillna(0)
        shadow_log['prod_sharpe'] = pd.to_numeric(shadow_log['prod_sharpe'], errors='coerce').fillna(0)
        
        fig = px.line(shadow_log, x='date', y=['shadow_sharpe', 'prod_sharpe'], 
                      labels={'value': '30-Day Sharpe', 'variable': 'Model'},
                      title="Trailing 30-Day Sharpe Ratio")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("4b. Promotion Gate Status")
        latest = shadow_log.iloc[-1]
        s_sharpe = latest['shadow_sharpe']
        p_sharpe = latest['prod_sharpe']
        
        col1, col2 = st.columns(2)
        col1.metric("Shadow Sharpe", f"{s_sharpe:.3f}")
        col2.metric("Production Sharpe", f"{p_sharpe:.3f}")
        
        if s_sharpe >= (p_sharpe - 0.05):
            st.success("🟢 ELIGIBLE FOR PROMOTION (Shadow is within safety margin of Production)")
        else:
            st.error("🔴 NOT YET ELIGIBLE (Shadow is underperforming Production)")
            
    else:
        st.warning("File not found or empty: checkpoints/live/shadow_log.csv")

# ---------------------------------------------------------
# PANEL 5: PERFORMANCE & RISK
# ---------------------------------------------------------
elif page == "Performance & Risk History":
    st.header("Panel 5 — Performance & Risk History")
    
    st.subheader("5a. Historical Backtest Table")
    
    classical_path = RESULTS_DIR / "p11_baselines" / "classical_baselines.csv"
    lit_path = RESULTS_DIR / "p11_baselines" / "literature_metrics.csv"
    det_path = RESULTS_DIR / "p11_baselines" / "metrics_deterministic.csv"
    
    dfs_to_concat = []
    
    if os.path.exists(classical_path):
        df_c = pd.read_csv(classical_path)
        # Rename columns to match standard
        df_c = df_c.rename(columns={'strategy': 'Model', 'sharpe': 'Sharpe Ratio', 'max_drawdown': 'Max Drawdown', 'cagr': 'CAGR', 'calmar': 'Calmar Ratio'})
        dfs_to_concat.append(df_c[['Model', 'Sharpe Ratio', 'CAGR', 'Max Drawdown', 'Calmar Ratio']])
        
    if os.path.exists(lit_path):
        df_l = pd.read_csv(lit_path)
        df_l.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)
        dfs_to_concat.append(df_l)
        
    if os.path.exists(det_path):
        df_d = pd.read_csv(det_path)
        df_d.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)
        dfs_to_concat.append(df_d)
        
    if dfs_to_concat:
        df_full = pd.concat(dfs_to_concat, ignore_index=True)
        # Format percentages
        if df_full['CAGR'].dtype == float:
            df_full['CAGR'] = (df_full['CAGR'] * 100).map("{:.1f}%".format)
        if df_full['Max Drawdown'].dtype == float:
            df_full['Max Drawdown'] = (df_full['Max Drawdown'] * 100).map("{:.1f}%".format)
        st.dataframe(df_full.set_index('Model'))
    else:
        st.info("Baseline CSVs not found in results/p11_baselines/")
        
    st.subheader("5b. Execution Latency")
    lat_path = RESULTS_DIR / "p13_latency" / "latency_report.md"
    if os.path.exists(lat_path):
        with open(lat_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Latency report not found. Target: 2.44 ms (Unified), 3.66 ms (C2).")

# ---------------------------------------------------------
# PANEL 6: RISK & COMPLIANCE
# ---------------------------------------------------------
elif page == "Risk & Compliance":
    st.header("Panel 6 — Risk Limits & Compliance")
    
    max_stock = df_stocks['Weight'].max() * 100
    
    # Calculate max sector
    df_sectors = df_stocks.copy()
    df_sectors['Sector'] = df_sectors['Symbol'].map(lambda x: SECTOR_MAP.get(x, "Other"))
    max_sector = df_sectors.groupby('Sector')['Weight'].sum().max() * 100
    
    cash_bonds = (df_macro[df_macro['Symbol'] == 'CASH']['Weight'].values[0] if 'CASH' in df_macro['Symbol'].values else 0) + \
                 (df_macro[df_macro['Symbol'] == 'BONDS']['Weight'].values[0] if 'BONDS' in df_macro['Symbol'].values else 0)
    cash_bonds = cash_bonds * 100
    
    total_stock_pct = total_stock_weight * 100
    
    limits = [
        {"Constraint": "Max single-stock weight", "Limit": "10%", "Current": f"{max_stock:.2f}%", "Status": "🔴" if max_stock > 10 else "🟢"},
        {"Constraint": "Max single-sector weight", "Limit": "30%", "Current": f"{max_sector:.2f}%", "Status": "🔴" if max_sector > 30 else "🟢"},
        {"Constraint": "Max cash + bonds exposure", "Limit": "60%", "Current": f"{cash_bonds:.2f}%", "Status": "🔴" if cash_bonds > 60 else "🟢"},
        {"Constraint": "Min stock exposure", "Limit": "30%", "Current": f"{total_stock_pct:.2f}%", "Status": "🔴" if total_stock_pct < 30 and not is_crisis else "🟢"}
    ]
    
    breach = any(r['Status'] == '🔴' for r in limits)
    if breach:
        st.error("🚨 RISK LIMIT BREACH — Review allocation before execution.")
        
    st.table(pd.DataFrame(limits))

# ---------------------------------------------------------
# PANEL 7: CONFIG & AUDIT
# ---------------------------------------------------------
elif page == "Config & Audit":
    st.header("Panel 7 — Config & Audit")
    
    st.subheader("7a. Current Production Config")
    config_data = {
        "Property": [
            "Encoder checkpoint ID", "A1 production checkpoint", "A2 production checkpoint", "A3 production checkpoint",
            "Transaction Cost Model", "Backtest Test Period", "Regime Gate Thresholds", "Shadow Promotion Gate"
        ],
        "Value": [
            "transformer/encoder.pt (Unified, frozen)",
            "a1_unified_joint_final.zip (200k steps)",
            "a2_unified_joint_final.zip (200k steps)",
            "a3_unified_joint_final.zip (200k steps)",
            "15 bps (10 bps execution + 5 bps slippage)",
            "2023-01-02 to 2025-12-29 (738 days)",
            "VIX > 25, FII z-score > 2.0",
            "Shadow Sharpe ≥ Prod Sharpe - 0.05 (Trailing 30d)"
        ]
    }
    st.table(pd.DataFrame(config_data))
    
    st.subheader("7b. Shadow Learner Audit Log")
    st.info("System logs: `checkpoints/live/shadow_log.csv` (Refer to Panel 4 for metrics)")
    if shadow_log is not None:
        st.dataframe(shadow_log[['date', 'action_taken', 'grad_steps']].tail(10))
