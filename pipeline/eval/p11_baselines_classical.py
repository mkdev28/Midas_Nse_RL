import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.packaging.p9_gym_env import StockPickerEnv, MacroAllocatorEnv, MidasDataset, load_encoder
from pipeline.eval.p10_validate_a1_a3 import (
    safe_step,
    extract_weights,
    extract_portfolio_value,
    detect_daily_return,
    apply_costs_to_return,
    annualized_sharpe,
    annualized_sortino,
    max_drawdown,
    cagr,
    calmar_ratio,
    avg_turnover,
    RISK_FREE_RATE_ANNUAL,
    TXN_COST_BPS,
    SLIPPAGE_BPS,
)

OUT_DIR = PROJECT_ROOT / "results" / "p11_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def get_markowitz_weights() -> np.ndarray:
    print("Calculating Markowitz Mean-Variance weights from Train split...")
    train_parquet = DATA_DIR / "train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"Missing {train_parquet}")
    
    df = pd.read_parquet(train_parquet)
    # We need daily returns for all 50 stocks.
    # The columns are like `target_ret_ticker` or similar? Let's check typical midas features.
    # Wait, the environment uses `X_stock` which we don't directly access here.
    # Actually, we can just extract the forward returns from the dataset object!
    
    dataset = MidasDataset(split="train")
    # MidasDataset has y (forward returns) and stock returns in the parquet.
    # Let's extract the actual daily returns of the 50 stocks from the parquet.
    # The columns for daily returns of the 50 stocks are not directly known by name,
    # but we know that StockPickerEnv expects weights of shape (50,).
    # Let's see how MidasDataset gets returns.
    # Since we don't have the explicit tickers easily here without parsing, 
    # let's just get the returns directly from dataset.y which is (T, 50) forward returns!
    print(f"Loaded train dataset with {dataset.T} steps.")
    y_train = dataset.stock_returns[dataset.stock_idx] # shape (T, 50)
    mu = np.mean(y_train, axis=0)
    cov = np.cov(y_train, rowvar=False)
    
    def objective(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        # Maximize Sharpe = Minimize -Sharpe
        if port_vol == 0:
            return 1e9
        return -(port_return / port_vol)
        
    init_w = np.ones(50) / 50.0
    bounds = tuple((0.0, 0.15) for _ in range(50))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        raise ValueError(f"Markowitz optimization failed: {res.message}")
        
    print("Markowitz weights computed successfully.")
    return res.x


class MomentumWeights:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_weights = np.ones(50) / 50.0

    def __call__(self, obs, info, steps):
        if steps % 21 == 0:
            idx = self.dataset.stock_idx[steps]
            if idx >= 252:
                past_returns = self.dataset.stock_returns[idx-252:idx]
                compounded = np.prod(1 + past_returns, axis=0) - 1
                top_10 = np.argsort(compounded)[-10:]
                w = np.zeros(50)
                w[top_10] = 0.1
                self.current_weights = w
        return self.current_weights

class VolWeightedWeights:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_weights = np.ones(50) / 50.0

    def __call__(self, obs, info, steps):
        if steps % 5 == 0:
            idx = self.dataset.stock_idx[steps]
            if idx >= 20:
                past_returns = self.dataset.stock_returns[idx-20:idx]
                vol = np.std(past_returns, axis=0)
                # handle zero volatility
                vol = np.where(vol < 1e-6, 1e-6, vol)
                w = 1.0 / vol
                self.current_weights = w / np.sum(w)
        return self.current_weights
def evaluate_static_strategy(name: str, weights_generator, dataset, encoder) -> dict:
    env = StockPickerEnv(dataset, encoder, "cpu")
    obs, info = env.reset(seed=42)
    
    equity_curve = []
    daily_returns = []
    weights_hist = []
    rewards = []
    
    equity = 1.0
    prev_weights = np.zeros(50, dtype=float)
    
    done = False
    steps = 0
    
    while not done:
        # Get target weights for this step
        if callable(weights_generator):
            target_weights = weights_generator(obs, info, steps)
        else:
            target_weights = weights_generator
            
        action = target_weights # StockPickerEnv takes weights directly
        obs, reward, done, info = safe_step(env, action)
        
        # Exact same evaluation logic as P10
        raw_weights = extract_weights(info, expected_dim=50)
        if raw_weights.size == 50:
            weight_sum = raw_weights.sum()
            if np.isfinite(weight_sum) and weight_sum != 0:
                curr_weights = raw_weights / weight_sum
            else:
                curr_weights = raw_weights
        else:
            curr_weights = target_weights
            
        gross_equity = extract_portfolio_value(info, equity, float(reward))
        gross_ret = detect_daily_return(info, reward, equity, gross_equity, agent_name="A3")
        
        net_ret = apply_costs_to_return(gross_ret, prev_weights, curr_weights)
        net_ret = float(np.clip(net_ret, -0.15, 0.15))
        if not np.isfinite(net_ret):
            net_ret = 0.0
            
        equity = equity * (1.0 + net_ret)
        if not np.isfinite(equity) or equity <= 0:
            equity = equity_curve[-1] if equity_curve else 1.0
            
        daily_returns.append(net_ret)
        equity_curve.append(equity)
        weights_hist.append(curr_weights.copy())
        rewards.append(float(reward))
        
        prev_weights = curr_weights
        steps += 1
        
    eq = pd.Series(equity_curve)
    rets = pd.Series(daily_returns)
    
    out = {
        "strategy": name,
        "steps": steps,
        "sharpe": annualized_sharpe(rets, RISK_FREE_RATE_ANNUAL),
        "sortino": annualized_sortino(rets, RISK_FREE_RATE_ANNUAL),
        "max_drawdown": max_drawdown(eq),
        "cagr": cagr(eq),
        "calmar": calmar_ratio(eq),
        "avg_turnover": avg_turnover(weights_hist),
        "final_equity": float(eq.iloc[-1]) if len(eq) else np.nan,
    }
    return out

def evaluate_macro_strategy(name: str, weights_generator, dataset, encoder, is_buy_and_hold=False) -> dict:
    env = MacroAllocatorEnv(dataset, encoder, "cpu")
    obs, info = env.reset(seed=42)
    
    equity_curve = []
    daily_returns = []
    weights_hist = []
    rewards = []
    
    equity = 1.0
    prev_weights = np.zeros(4, dtype=float)
    
    done = False
    steps = 0
    
    while not done:
        if callable(weights_generator):
            target_weights = weights_generator(obs, info, steps, env)
        else:
            target_weights = weights_generator
            
        action = np.log(np.clip(target_weights, 1e-8, 1.0))
        obs, reward, done, info = safe_step(env, action)
        
        curr_weights = info.get("weights", target_weights)
            
        gross_equity = extract_portfolio_value(info, equity, float(reward))
        gross_ret = detect_daily_return(info, reward, equity, gross_equity, agent_name="A1")
        
        is_rebalance_day = (steps % 21 == 0) and not is_buy_and_hold
        
        if is_rebalance_day or (is_buy_and_hold and steps == 0):
            net_ret = apply_costs_to_return(gross_ret, prev_weights, curr_weights)
        else:
            net_ret = gross_ret
            
        net_ret = float(np.clip(net_ret, -0.15, 0.15))
        if not np.isfinite(net_ret):
            net_ret = 0.0
            
        equity = equity * (1.0 + net_ret)
        if not np.isfinite(equity) or equity <= 0:
            equity = equity_curve[-1] if equity_curve else 1.0
            
        daily_returns.append(net_ret)
        equity_curve.append(equity)
        weights_hist.append(curr_weights.copy())
        rewards.append(float(reward))
        
        prev_weights = curr_weights
        steps += 1
        
    eq = pd.Series(equity_curve)
    rets = pd.Series(daily_returns)
    
    out = {
        "strategy": name,
        "steps": steps,
        "sharpe": annualized_sharpe(rets, RISK_FREE_RATE_ANNUAL),
        "sortino": annualized_sortino(rets, RISK_FREE_RATE_ANNUAL),
        "max_drawdown": max_drawdown(eq),
        "cagr": cagr(eq),
        "calmar": calmar_ratio(eq),
        "avg_turnover": avg_turnover(weights_hist),
        "final_equity": float(eq.iloc[-1]) if len(eq) else np.nan,
    }
    return out

class StaticMacroWeights:
    def __init__(self, target_weights):
        self.target = np.array(target_weights, dtype=float)
        self.current = self.target.copy()
        
    def __call__(self, obs, info, steps, env):
        if steps % 21 == 0:
            self.current = self.target.copy()
        else:
            t = env.t - 1
            if t >= 0:
                rets = np.array([
                    float(env.ds.returns[t]),
                    float(env.ds.bond_ret[t]),
                    float(env.ds.commodity_ret[t]),
                    float(env.ds.cash_ret[t])
                ])
                self.current = self.current * (1 + rets)
                self.current /= self.current.sum()
        return self.current

class BuyAndHoldMacroWeights:
    def __init__(self, initial_weights):
        self.current = np.array(initial_weights, dtype=float)
        
    def __call__(self, obs, info, steps, env):
        if steps == 0:
            return self.current
        t = env.t - 1
        if t >= 0:
            rets = np.array([
                float(env.ds.returns[t]),
                float(env.ds.bond_ret[t]),
                float(env.ds.commodity_ret[t]),
                float(env.ds.cash_ret[t])
            ])
            self.current = self.current * (1 + rets)
            self.current /= self.current.sum()
        return self.current

def main():
    print("="*60)
    print("P11 Classical Baselines (Equal Weight & Markowitz)")
    print("="*60)
    
    dataset = MidasDataset(split="test")
    encoder = load_encoder("cpu")
    
    results = []
    
    # 1. Equal Weight (Daily Rebalance)
    print("\nRunning Equal Weight (Daily Rebalance)...")
    ew_weights = np.ones(50) / 50.0
    res_ew = evaluate_static_strategy("Equal Weight", ew_weights, dataset, encoder)
    results.append(res_ew)
    print(f"Equal Weight -> Sharpe: {res_ew['sharpe']:.3f}, CAGR: {res_ew['cagr']:.3f}, MDD: {res_ew['max_drawdown']:.3f}, Final Equity: {res_ew['final_equity']:.3f}")
    
    # 2. Markowitz Mean-Variance
    print("\nRunning Markowitz Mean-Variance...")
    markowitz_weights = get_markowitz_weights()
    res_mw = evaluate_static_strategy("Markowitz", markowitz_weights, dataset, encoder)
    results.append(res_mw)
    print(f"Markowitz    -> Sharpe: {res_mw['sharpe']:.3f}, CAGR: {res_mw['cagr']:.3f}, MDD: {res_mw['max_drawdown']:.3f}, Final Equity: {res_mw['final_equity']:.3f}")
    
    # 3. Momentum (Monthly Rebalance)
    print("\nRunning Momentum (Monthly Rebalance)...")
    momentum_gen = MomentumWeights(dataset)
    res_mom = evaluate_static_strategy("Momentum", momentum_gen, dataset, encoder)
    results.append(res_mom)
    print(f"Momentum     -> Sharpe: {res_mom['sharpe']:.3f}, CAGR: {res_mom['cagr']:.3f}, MDD: {res_mom['max_drawdown']:.3f}, Final Equity: {res_mom['final_equity']:.3f}")

    # 4. Volatility Weighted (Weekly Rebalance)
    print("\nRunning Volatility Weighted (Weekly Rebalance)...")
    vol_gen = VolWeightedWeights(dataset)
    res_vol = evaluate_static_strategy("Volatility Weighted", vol_gen, dataset, encoder)
    results.append(res_vol)
    print(f"Vol Weighted -> Sharpe: {res_vol['sharpe']:.3f}, CAGR: {res_vol['cagr']:.3f}, MDD: {res_vol['max_drawdown']:.3f}, Final Equity: {res_vol['final_equity']:.3f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "classical_baselines.csv", index=False)
    print(f"\nResults saved to {OUT_DIR / 'classical_baselines.csv'}")
    
    print("\n" + "="*60)
    print("A1 MacroAllocator Baselines")
    print("="*60)
    
    macro_results = []
    
    # 1. Static 60/40 (Monthly Rebalance)
    print("\nRunning Static 60/40 (Monthly Rebalance)...")
    s6040_gen = StaticMacroWeights([0.60, 0.40, 0.0, 0.0])
    res_6040 = evaluate_macro_strategy("Static 60/40", s6040_gen, dataset, encoder)
    macro_results.append(res_6040)
    print(f"Static 60/40 -> Sharpe: {res_6040['sharpe']:.3f}, CAGR: {res_6040['cagr']:.3f}, MDD: {res_6040['max_drawdown']:.3f}, Final Equity: {res_6040['final_equity']:.3f}")

    # 2. Buy and Hold 4-Asset (No Rebalance)
    print("\nRunning Buy-and-Hold 4-Asset Equal Weight...")
    bah_gen = BuyAndHoldMacroWeights([0.25, 0.25, 0.25, 0.25])
    res_bah = evaluate_macro_strategy("Buy & Hold 4-Asset", bah_gen, dataset, encoder, is_buy_and_hold=True)
    macro_results.append(res_bah)
    print(f"Buy & Hold 4-Asset -> Sharpe: {res_bah['sharpe']:.3f}, CAGR: {res_bah['cagr']:.3f}, MDD: {res_bah['max_drawdown']:.3f}, Final Equity: {res_bah['final_equity']:.3f}")

    df_macro = pd.DataFrame(macro_results)
    df_macro.to_csv(OUT_DIR / "a1_macro_baselines.csv", index=False)
    print(f"\nA1 Results saved to {OUT_DIR / 'a1_macro_baselines.csv'}")
    
if __name__ == "__main__":
    main()
