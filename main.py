import os
import sys
import warnings
import time
import json
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from supabase import create_client, Client
import streamlit as st

warnings.filterwarnings("ignore")

# ==========================================
# 1. ROBUUSTE CONFIGURATIE (LOKAAL & GITHUB)
# ==========================================
SUPABASE_URL = None
SUPABASE_KEY = None

# Stap 1: Probeer Streamlit Secrets (voor lokaal/cloud gebruik)
try:
    if hasattr(st, "secrets") and "SUPABASE_URL" in st.secrets:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
        print("✅ Config geladen via Streamlit Secrets")
except (FileNotFoundError, KeyError, AttributeError):
    pass

# Stap 2: Fallback naar Environment Variables (voor GitHub Actions)
if not SUPABASE_URL:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    if SUPABASE_URL:
        print("✅ Config geladen via Environment Variables")

# Stap 3: Harde check - Stop als sleutels ontbreken
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ FATAL ERROR: Geen SUPABASE_URL of SUPABASE_KEY gevonden!")
    print("   -> Check je GitHub Repository Secrets of .streamlit/secrets.toml")
    sys.exit(1) # Dit zorgt voor een rode 'fail' in GitHub Actions

# Verbinding maken
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ FATAL ERROR: Kan geen verbinding maken met Supabase: {e}")
    sys.exit(1)

# ==========================================
# 2. DATA LOADING
# ==========================================
def get_full_market_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    
    def read_wiki(url, idx=0, attrs=None):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if attrs:
                return pd.read_html(r.text, attrs=attrs)[0]
            return pd.read_html(r.text)[idx]
        except Exception as e:
            print(f"⚠️ Warning: Kon data niet lezen van {url} ({e})")
            return pd.DataFrame()

    # Haal tickers op
    sp500 = read_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", attrs={"id": "constituents"})
    nasdaq = read_wiki("https://en.wikipedia.org/wiki/Nasdaq-100", idx=4)
    sp400 = read_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", idx=0)

    # Cleaning
    s500_list = [str(s).replace('.', '-') for s in sp500['Symbol']] if not sp500.empty else []
    
    # Nasdaq check (kolomnaam varieert soms)
    n_col = next((c for c in ['Ticker', 'Symbol'] if not nasdaq.empty and c in nasdaq.columns), None)
    nas_list = [str(s).replace('.', '-') for s in nasdaq[n_col]] if n_col else []
    
    s400_list = [str(s).replace('.', '-') for s in sp400['Symbol']] if not sp400.empty else []

    tickers = list(set(s500_list + nas_list + s400_list))
    return tickers, set(s500_list), set(nas_list), set(s400_list)

# ==========================================
# 3. ENGINE
# ==========================================
def run_engine():
    print(f"🚀 Start run op {datetime.now()}")
    
    try:
        tickers, s500_set, nas_set, s400_set = get_full_market_tickers()
    except Exception as e:
        print(f"❌ Error bij ophalen tickers: {e}")
        sys.exit(1)

    if not tickers:
        print("❌ Geen tickers gevonden. Stoppen.")
        sys.exit(1)

    print(f"📊 Analyse van {len(tickers)} aandelen gestart...")
    run_date = datetime.now().strftime("%Y-%m-%d")
    
    all_train = []
    current_scan = []

    # Loop door aandelen
    # Tip: Voor testen kun je tickers[:10] gebruiken
    for i, symbol in enumerate(tickers):
        if i % 50 == 0: print(f"   ... bezig met aandeel {i}/{len(tickers)}")
        
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
                
                # Check for NaNs in result
                res = [rsi.iloc[idx], vol.iloc[idx], vol_s.iloc[idx]]
                return [0 if np.isnan(x) else x for x in res]

            # 1. Bouw Training Set
            for i in range(1, 8):
                # We pakken samples verspreid over het jaar
                idx = -(i * 20 + 1)
                if abs(idx) >= len(hist): continue
                
                feats = f_base + get_tech_features(hist, idx)
                
                # Interne functie om rendement te berekenen
                def get_return(d):
                    if idx + d < 0:
                        p1 = hist['Close'].iloc[idx]
                        p2 = hist['Close'].iloc[idx+d]
                        return (p2 - p1) / p1
                    return None

                t2 = get_return(10) # 2 weken
                t4 = get_return(20) # 4 weken

                if t2 is not None and t4 is not None:
                    all_train.append({
                        "features": feats,
                        "t_2w": t2,
                        "t_4w": t4
                    })

            # 2. Huidige Data (voor voorspelling van morgen)
            curr_feats = f_base + get_tech_features(hist, -1)
            exch = "SP500" if symbol in s500_set else "NASDAQ" if symbol in nas_set else "SP400" if symbol in s400_set else "OTHER"
            
            current_scan.append({
                "run_date": run_date,
                "ticker": symbol,
                "exchange": exch,
                "price": round(hist['Close'].iloc[-1], 2),
                "features": curr_feats
            })
            
        except Exception as e:
            # Eén aandeel faalt? Geen probleem, ga door.
            continue

    # Check of we data hebben
    if not all_train or not current_scan:
        print("❌ Geen trainingsdata of scan data verzameld.")
        sys.exit(1)
    
    df_train = pd.DataFrame(all_train).dropna()
    df_scan = pd.DataFrame(current_scan)
    
    print(f"🧠 Training set grootte: {len(df_train)} samples")

    # Modellen Trainen
    print("🤖 Modellen trainen...")
    
    def predict(target_col):
        X = np.array(df_train['features'].tolist())
        y = df_train[target_col].values
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=60, max_depth=8, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        
        X_scan = np.array(df_scan['features'].tolist())
        preds = rf.predict(X_scan)
        
        # Confidence via tree variantie
        trees = np.array([t.predict(X_scan) for t in rf.estimators_])
        unc = np.std(trees, axis=0)
        p95 = np.percentile(unc, 95) if len(unc) > 0 else 1
        
        # Normalize confidence 0-100
        conf = np.clip(1 - unc / (p95 if p95 > 0 else 1), 0, 1) * 100
        
        return preds, conf

    # Voer voorspellingen uit
    try:
        df_scan["alpha_2w"], df_scan["confidence_2w"] = predict("t_2w")
        df_scan["alpha_4w"], df_scan["confidence_4w"] = predict("t_4w")
    except Exception as e:
        print(f"❌ Fout tijdens trainen/voorspellen: {e}")
        sys.exit(1)

    # Normaliseren (Z-score per exchange)
    for c in ["alpha_2w", "alpha_4w"]:
        df_scan[f"{c}_norm"] = df_scan.groupby("exchange")[c].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

    # Signaal bepalen
    def determine_signal(r):
        # Iets strengere eisen voor een signaal
        if r["alpha_2w_norm"] > 1.0 and r["confidence_2w"] > 70: return "LONG"
        if r["alpha_2w_norm"] < -1.0 and r["confidence_2w"] > 70: return "SHORT"
        return "NEUTRAL"

    df_scan["signal"] = df_scan.apply(determine_signal, axis=1)

    # =======================================================
    # DE OPLOSSING VOOR DE JSON/SUPABASE ERROR
    # =======================================================
    # Features kolom verwijderen (kan niet in DB)
    upload_df = df_scan.drop(columns=["features"])
    
    # Voorkom errors door infinite waardes te veranderen naar NaN
    upload_df = upload_df.replace([np.inf, -np.inf], np.nan)
    
    # Gebruik Pandas .to_json() om alle numpy types (float64, int64) correct te vertalen
    # naar native Python types. Dit vertaalt NaN automatisch naar null (None).
    json_records = upload_df.to_json(orient='records')
    upload_data = json.loads(json_records)
    # =======================================================
    
    print(f"☁️ Uploading {len(upload_data)} records naar Supabase...")
    
    batch_size = 100
    errors = 0
    
    for i in range(0, len(upload_data), batch_size):
        try:
            batch = upload_data[i:i+batch_size]
            supabase.table('stock_predictions').upsert(batch, on_conflict='Run_Date,Ticker').execute()
        except Exception as e:
            print(f"⚠️ Error batch {i}: {e}")
            errors += 1
            if errors > 5:
                print("❌ Te veel upload errors. Aborting.")
                sys.exit(1)

    print("✅ Klaar! Run succesvol afgerond.")

if __name__ == "__main__":
    run_engine()
