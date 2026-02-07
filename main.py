import os
import warnings
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG & SUPABASE SETUP
# ==========================================
SUPABASE_URL = "https://ibffbjlvibkisbzecfkn.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_KEY:
    print("LET OP: Geen SUPABASE_KEY gevonden in environment variables.")
    exit()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# DATA LOADING
# ==========================================
def get_full_market_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    
    def read_wiki(url, idx=0, attrs=None):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if attrs:
                return pd.read_html(r.text, attrs=attrs)[0]
            return pd.read_html(r.text)[idx]
        except:
            return pd.DataFrame()

    # Haal tickers op
    sp500 = read_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", attrs={"id": "constituents"})
    nasdaq = read_wiki("https://en.wikipedia.org/wiki/Nasdaq-100", idx=4)
    sp400 = read_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", idx=0)

    # Cleaning
    s500_list = [s.replace('.', '-') for s in sp500['Symbol']] if not sp500.empty else []
    
    n_col = next((c for c in ['Ticker', 'Symbol'] if not nasdaq.empty and c in nasdaq.columns), None)
    nas_list = [s.replace('.', '-') for s in nasdaq[n_col]] if n_col else []
    
    s400_list = [s.replace('.', '-') for s in sp400['Symbol']] if not sp400.empty else []

    tickers = list(set(s500_list + nas_list + s400_list))
    return tickers, set(s500_list), set(nas_list), set(s400_list)

# ==========================================
# ENGINE
# ==========================================
def run_engine():
    tickers, s500_set, nas_set, s400_set = get_full_market_tickers()
    run_date = datetime.now().strftime("%Y-%m-%d")

    if not tickers:
        print("Geen tickers gevonden.")
        return

    print(f"Analyse van {len(tickers)} aandelen gestart...")
    
    all_train = []
    current_scan = []

    # Loop door aandelen
    for symbol in tickers:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2y")
            
            if len(hist) < 260: continue

            # Features berekenen
            info = stock.info
            f_base = [
                info.get('trailingPE', 20) or 20,
                info.get('returnOnEquity', 0.1) or 0.1,
                info.get('debtToEquity', 100) or 100,
                (info.get('freeCashflow', 0) or 0) / (info.get('marketCap', 1) or 1)
            ]

            def get_tech_features(df, idx):
                delta = df['Close'].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = -delta.clip(upper=0).rolling(14).mean()
                
                # Veilige RSI
                rs = gain / loss
                rs = rs.replace([np.inf, -np.inf], 0).fillna(0)
                rsi = 100 - (100 / (1 + rs))
                
                vol = df['Close'].pct_change().rolling(20).std()
                vol_s = df['Volume'] / df['Volume'].rolling(20).mean()
                return [rsi.iloc[idx], vol.iloc[idx], vol_s.iloc[idx]]

            # 1. Bouw Training Set
            for i in range(1, 8):
                idx = -(i * 20 + 1)
                if abs(idx) >= len(hist): continue
                
                feats = f_base + get_tech_features(hist, idx)
                
                def get_return(d):
                    if idx + d < 0:
                        p1 = hist['Close'].iloc[idx]
                        p2 = hist['Close'].iloc[idx+d]
                        return (p2 - p1) / p1
                    return None

                all_train.append({
                    "features": feats,
                    "t_2w": get_return(10),
                    "t_4w": get_return(20)
                })

            # 2. Huidige Data
            curr_feats = f_base + get_tech_features(hist, -1)
            exch = "SP500" if symbol in s500_set else "NASDAQ" if symbol in nas_set else "SP400" if symbol in s400_set else "OTHER"
            
            current_scan.append({
                "run_date": run_date,
                "ticker": symbol,
                "exchange": exch,
                "price": round(hist['Close'].iloc[-1], 2),
                "features": curr_feats
            })
            
        except Exception:
            continue

    # DataFrames maken
    if not all_train or not current_scan: return
    
    df_train = pd.DataFrame(all_train).dropna()
    df_scan = pd.DataFrame(current_scan)

    # Modellen Trainen
    print("Modellen trainen...")
    
    def predict(target):
        X = np.array(df_train['features'].tolist())
        y = df_train[target].values
        
        # Snelle Random Forest
        rf = RandomForestRegressor(n_estimators=60, max_depth=8, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        
        X_scan = np.array(df_scan['features'].tolist())
        preds = rf.predict(X_scan)
        
        # Confidence
        trees = np.array([t.predict(X_scan) for t in rf.estimators_])
        unc = np.std(trees, axis=0)
        p95 = np.percentile(unc, 95) if len(unc) > 0 else 1
        conf = np.clip(1 - unc / (p95 if p95 > 0 else 1), 0, 1) * 100
        
        return preds, conf

    df_scan["alpha_2w"], df_scan["confidence_2w"] = predict("t_2w")
    df_scan["alpha_4w"], df_scan["confidence_4w"] = predict("t_4w")

    # Normaliseren
    for c in ["alpha_2w", "alpha_4w"]:
        df_scan[f"{c}_norm"] = df_scan.groupby("exchange")[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    # Signaal
    def signal(r):
        if r["alpha_2w_norm"] > 0.8 and r["confidence_2w"] > 75: return "LONG"
        if r["alpha_2w_norm"] < -0.8 and r["confidence_2w"] > 75: return "SHORT"
        return "NEUTRAL"

    df_scan["signal"] = df_scan.apply(signal, axis=1)

    # Uploaden
    upload_data = df_scan.drop(columns=["features"]).replace({np.nan: None}).to_dict(orient='records')
    
    print(f"Uploading {len(upload_data)} records...")
    
    batch = 100
    for i in range(0, len(upload_data), batch):
        try:
            supabase.table('stock_predictions').upsert(upload_data[i:i+batch]).execute()
        except Exception as e:
            print(f"Error batch {i}: {e}")

    print("Klaar!")

if __name__ == "__main__":
    run_engine()
