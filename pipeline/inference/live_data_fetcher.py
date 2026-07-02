import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import joblib

LIVE_DIR = Path("data/live")
LIVE_DIR.mkdir(parents=True, exist_ok=True)

try:
    meta = joblib.load("data/processed/stock_features_meta.pkl")
    NIFTY_SYMBOLS = list(meta['tickers'])
except Exception:
    # Fallback to the known list
    NIFTY_SYMBOLS = ['ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'M&M.NS']

def fetch_live_raw_data(tickers, end_date: datetime, window_days=100):
    start_date = end_date - timedelta(days=window_days)
    date_str = end_date.strftime("%Y-%m-%d")
    cache_dir = LIVE_DIR / date_str
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Prices
    prices_path = cache_dir / "prices.csv"
    if prices_path.exists():
        prices_df = pd.read_csv(prices_path, index_col=0, header=[0, 1], parse_dates=True)
    else:
        end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        start_str = start_date.strftime("%Y-%m-%d")
        prices_df = yf.download(tickers, start=start_str, end=end_str, progress=False)
        prices_df.to_csv(prices_path)
    
    prices_df = prices_df.loc[prices_df.index <= pd.Timestamp(end_date)]
    
    # 2. Fetch Macro (We mock a few raw streams that drive the 46 dims)
    # The true 46 offline features come from a much larger offline pipeline.
    # We'll fetch standard proxies to reconstruct the input structure.
    macro_path = cache_dir / "macro.csv"
    macro_tickers = ["^INDIAVIX", "INR=X", "^GSPC", "^IRX", "CL=F", "GC=F"]
    if macro_path.exists():
        macro_df = pd.read_csv(macro_path, index_col=0, header=[0, 1], parse_dates=True)
    else:
        end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
        start_str = start_date.strftime("%Y-%m-%d")
        macro_df = yf.download(macro_tickers, start=start_str, end=end_str, progress=False)
        macro_df.to_csv(macro_path)
        
    macro_df = macro_df.loc[macro_df.index <= pd.Timestamp(end_date)]
    
    # 3. Fetch News (Mocked as neutral deterministic zero vector for A2 shape)
    # Replaces random normal with deterministic zeros per deep research instruction
    news_df = pd.DataFrame(index=prices_df.index, columns=["sentiment_1", "sentiment_2", "sentiment_3"])
    news_df[:] = 0.0
    
    return prices_df, macro_df, news_df

def compute_live_features(prices_df, macro_df, news_df, end_date, window_size=60):
    if len(prices_df) < window_size:
        raise ValueError(f"Not enough data to form a {window_size}-day window.")
        
    # Take the last `window_size` days strictly up to end_date
    prices_window = prices_df.iloc[-window_size:]
    macro_window = macro_df.iloc[-window_size:]
    news_window = news_df.iloc[-window_size:]
    
    # -- 1. Stock Features --
    num_stocks = len(NIFTY_SYMBOLS)
    # Note: Output shape MUST be (window_size, num_stocks, 12) for the P9 dataset structure!
    # I'll build it as (num_stocks, window_size, 12) then transpose to (window_size, num_stocks, 12)
    X_stock_raw = np.zeros((num_stocks, window_size, 12), dtype=np.float32)
    
    for i, sym in enumerate(NIFTY_SYMBOLS):
        if sym not in prices_df.columns.levels[1] if isinstance(prices_df.columns, pd.MultiIndex) else prices_df.columns:
            continue
            
        try:
            if isinstance(prices_df.columns, pd.MultiIndex):
                close = prices_df['Close'][sym]
                high = prices_df['High'][sym]
                low = prices_df['Low'][sym]
                open_p = prices_df['Open'][sym]
                vol = prices_df['Volume'][sym]
            else:
                close = prices_df['Close']
                high = prices_df['High']
                low = prices_df['Low']
                open_p = prices_df['Open']
                vol = prices_df['Volume']
                
            df_sym = pd.DataFrame({'Open': open_p, 'High': high, 'Low': low, 'Close': close, 'Volume': vol})
            
            # Exact 12 Tech Features in Order:
            # ['ema_5', 'ema_20', 'rsi_14', 'macd', 'macd_hist', 'atr_14', 'bb_width', 'obv', 'mfi_14', 'daily_return', 'log_return', 'rolling_sharpe_30']
            
            df_sym['ema_5'] = df_sym['Close'].ewm(span=5).mean()
            df_sym['ema_20'] = df_sym['Close'].ewm(span=20).mean()
            
            delta = df_sym['Close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            df_sym['rsi_14'] = 100 - (100 / (1 + rs))
            
            ema12 = df_sym['Close'].ewm(span=12).mean()
            ema26 = df_sym['Close'].ewm(span=26).mean()
            df_sym['macd'] = ema12 - ema26
            df_sym['macd_signal'] = df_sym['macd'].ewm(span=9).mean()
            df_sym['macd_hist'] = df_sym['macd'] - df_sym['macd_signal']
            
            tr = pd.concat([df_sym['High'] - df_sym['Low'], 
                            (df_sym['High'] - df_sym['Close'].shift()).abs(),
                            (df_sym['Low'] - df_sym['Close'].shift()).abs()], axis=1).max(axis=1)
            df_sym['atr_14'] = tr.rolling(14).mean()
            
            sma20 = df_sym['Close'].rolling(20).mean()
            std20 = df_sym['Close'].rolling(20).std()
            df_sym['bb_width'] = (4 * std20) / sma20
            
            obv = (np.sign(df_sym['Close'].diff()) * df_sym['Volume']).fillna(0).cumsum()
            df_sym['obv'] = obv
            
            typical_price = (df_sym['High'] + df_sym['Low'] + df_sym['Close']) / 3
            raw_money_flow = typical_price * df_sym['Volume']
            pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0), index=df_sym.index).rolling(14).sum()
            neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0), index=df_sym.index).rolling(14).sum()
            df_sym['mfi_14'] = 100 - (100 / (1 + pos_flow / (neg_flow + 1e-8)))
            
            df_sym['daily_return'] = df_sym['Close'].pct_change(1)
            df_sym['log_return'] = np.log(df_sym['Close'] / df_sym['Close'].shift(1))
            df_sym['rolling_sharpe_30'] = (df_sym['daily_return'].rolling(30).mean() / (df_sym['daily_return'].rolling(30).std() + 1e-8)) * np.sqrt(252)
            
            features_12 = df_sym[['ema_5', 'ema_20', 'rsi_14', 'macd', 'macd_hist', 'atr_14', 'bb_width', 'obv', 'mfi_14', 'daily_return', 'log_return', 'rolling_sharpe_30']]
            features_12 = features_12.bfill().fillna(0)
            
            df_sym_window = features_12.iloc[-window_size:]
            X_stock_raw[i, :, :] = df_sym_window.values
            
        except Exception as e:
            pass
            
    # Apply historical scalers
    try:
        means = np.load("data/processed/stock_features_means.npy")
        stds = np.load("data/processed/stock_features_stds.npy")
        X_stock_raw = (X_stock_raw - means) / (stds + 1e-8)
    except FileNotFoundError:
        print("Warning: Historical stock scalers not found.")
        
    # Output shape should match X_stock in envs: (window_size, 50, 12)
    X_stock = np.transpose(X_stock_raw, (1, 0, 2))
        
    # -- 2. Macro Features --
    # In training, macro is 46-dimensional. We mock missing dimensions using the raw data.
    X_macro = np.zeros((window_size, 46), dtype=np.float32)
    if isinstance(macro_window.columns, pd.MultiIndex):
        close_macro = macro_window['Close'].bfill().fillna(0).values
    else:
        close_macro = macro_window.bfill().fillna(0).values
        
    cols_to_fill = min(46, close_macro.shape[1])
    X_macro[:, :cols_to_fill] = close_macro[:, :cols_to_fill]
    
    try:
        scaler = joblib.load("data/processed/scaler.pkl")
        if scaler.n_features_in_ == 46:
            X_macro = scaler.transform(X_macro)
    except FileNotFoundError:
        print("Warning: scaler.pkl not found.")
        
    # -- 3. Sentiment Features --
    # Return full window for completeness (window_size, 3)
    X_sentiment = news_window.values.astype(np.float32)
    
    return X_stock, X_macro, X_sentiment

if __name__ == "__main__":
    today = datetime.today()
    prices, macro, news = fetch_live_raw_data(NIFTY_SYMBOLS, today, window_days=100)
    X_stock, X_macro, X_sentiment = compute_live_features(prices, macro, news, today, window_size=60)
    
    print("Stock features shape:", X_stock.shape)
    print("Macro features shape:", X_macro.shape)
    print("Sentiment features shape:", X_sentiment.shape)
