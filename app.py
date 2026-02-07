import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & SETUP
# ==========================================
st.set_page_config(page_title="Stock Sweet Spots", layout="wide")

# We halen de secrets veilig op uit Streamlit Cloud instellingen
# Als je lokaal test, maakt hij een bestand .streamlit/secrets.toml nodig,
# of je vult ze hieronder tijdelijk hard in (maar niet committen naar GitHub!).
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except FileNotFoundError:
    st.error("Geen secrets gevonden. Stel SUPABASE_URL en SUPABASE_KEY in.")
    st.stop()

@st.cache_data(ttl=600)
def load_data():
    try:
        # Maak verbinding
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Haal data op
        response = supabase.table('stock_predictions').select("*").execute()
        
        # Check of de lijst leeg is
        if not response.data:
            return pd.DataFrame()

        # Maak dataframe
        df = pd.DataFrame(response.data)

        # --- DE OPLOSSING ---
        # We zetten ALLE kolomnamen om naar kleine letters.
        # Dus: 'Run_Date' wordt 'run_date', 'Ticker' wordt 'ticker', etc.
        df.columns = df.columns.str.lower()
        # --------------------

        # Nu kunnen we veilig verder met kleine letters
        if 'run_date' in df.columns:
            df['run_date'] = pd.to_datetime(df['run_date'])
            df = df.sort_values('run_date')
        
        return df

    except Exception as e:
        st.error(f"Er ging iets mis: {e}")
        return pd.DataFrame()

data = load_data()

st.title("🎯 Sweet Spot Monitor")

if data.empty:
    st.warning("Nog geen data gevonden in Supabase. Wacht op de eerste run (maandagochtend).")
    st.stop()

# ==========================================
# 2. REKENFUNCTIE (DE LOGICA)
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    """
    Zoekt signalen en berekent het behaalde maximum in de periode erna.
    """
    results = []
    
    # Filter: Confidence > 70% EN Alpha Norm > 1
    # (Pas de 1.0 aan naar 0.75 of 0.8 als je meer resultaten wilt zien)
    signals = df[(df[conf_col] > 70) & (df[alpha_col] > 1)].copy()
    
    for idx, row in signals.iterrows():
        ticker = row['ticker']
        start_date = row['run_date']
        start_price = row['price']
        
        # De periode waarin we zoeken naar een hogere prijs
        end_date = start_date + timedelta(days=lookahead_days)
        
        # Haal data op van dit aandeel:
        # - NA de signaal datum
        # - MAAR VOOR (of op) de einddatum
        future_mask = (
            (df['ticker'] == ticker) & 
            (df['run_date'] > start_date) & 
            (df['run_date'] <= end_date)
        )
        future_data = df[future_mask]
        
        # Bepaal het (voorlopige) maximum
        if not future_data.empty:
            max_price_seen = future_data['price'].max()
            days_data_available = (future_data['run_date'].max() - start_date).days
        else:
            # Nog geen data na het signaal (bijv. signaal was gisteren)
            max_price_seen = start_price
            days_data_available = 0
            
        pct_diff = ((max_price_seen - start_price) / start_price) * 100
        
        # Status bepalen voor de gebruiker
        if days_data_available >= lookahead_days:
            status = "🏁 Voltooid"
        elif days_data_available == 0:
            status = "🆕 Nieuw"
        else:
            status = f"⏳ Dag {days_data_available}/{lookahead_days}"

        results.append({
            "Datum": start_date.strftime('%Y-%m-%d'),
            "Ticker": ticker,
            "Prijs (Start)": start_price,
            "Conf": row[conf_col],
            "Alpha": row[alpha_col],
            "Max Prijs (Tot nu)": max_price_seen,
            "Winst %": pct_diff,
            "Status": status
        })
        
    return pd.DataFrame(results)

# ==========================================
# 3. VISUALISATIE (GRAFIEK)
# ==========================================
st.subheader("🔎 Analyse per Aandeel")

# Selectbox met alle beschikbare tickers
unique_tickers = data['ticker'].unique()
ticker = st.selectbox("Selecteer aandeel om te analyseren:", unique_tickers)

if ticker:
    subset = data[data['ticker'] == ticker]
    
    fig = go.Figure()

    # Linker Y-As: Alpha (Volle Lijnen)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name='Alpha Norm 2W', line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_4w_norm'],
        name='Alpha Norm 4W', line=dict(color='cyan', width=2)
    ))

    # Rechter Y-As: Confidence (Stippellijnen)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_2w'],
        name='Conf 2W', line=dict(color='red', width=2, dash='dot'),
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_4w'],
        name='Conf 4W', line=dict(color='orange', width=2, dash='dot'),
        yaxis='y2'
    ))

    # Layout voor dubbele as
    fig.update_layout(
        title=f"Signaalverloop {ticker}",
        xaxis_title="Datum",
        yaxis=dict(
            title="Alpha Norm (Standaardafwijking)",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title="Confidence (%)",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================
# 4. DE TABELLEN (SWEET SPOTS)
# ==========================================

col1, col2 = st.columns(2)

# --- TABEL 1: 2 WEKEN PREDICTIE (Kijk 21 dagen vooruit) ---
with col1:
    st.markdown("### ⚡ Sweet Spot: 2 Weken")
    st.markdown("*Max prijs in de **3 weken** (21 dagen) na signaal*")
    
    # Hier roepen we de functie aan met lookahead=21
    df_2w = get_sweet_spots(data, lookahead_days=21, alpha_col='alpha_2w_norm', conf_col='confidence_2w')
    
    if not df_2w.empty:
        # Bereken gemiddelde winst van de lijst
        avg_gain = df_2w['Winst %'].mean()
        st.metric("Gemiddelde 'Max Winst' van signalen", f"{avg_gain:.2f}%")
        
        # Toon tabel met kleuren
        st.dataframe(
            df_2w.style.format({
                "Prijs (Start)": "{:.2f}",
                "Max Prijs (Tot nu)": "{:.2f}",
                "Conf": "{:.1f}%",
                "Alpha": "{:.2f}",
                "Winst %": "{:.2f}%"
            }).background_gradient(subset=['Winst %'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Geen signalen gevonden (Alpha > 1 & Conf > 70).")

# --- TABEL 2: 4 WEKEN PREDICTIE (Kijk 28 dagen vooruit) ---
with col2:
    st.markdown("### 🔮 Sweet Spot: 4 Weken")
    st.markdown("*Max prijs in de **4 weken** (28 dagen) na signaal*")
    
    # Hier roepen we de functie aan met lookahead=28
    df_4w = get_sweet_spots(data, lookahead_days=28, alpha_col='alpha_4w_norm', conf_col='confidence_4w')
    
    if not df_4w.empty:
        avg_gain = df_4w['Winst %'].mean()
        st.metric("Gemiddelde 'Max Winst' van signalen", f"{avg_gain:.2f}%")
        
        st.dataframe(
            df_4w.style.format({
                "Prijs (Start)": "{:.2f}",
                "Max Prijs (Tot nu)": "{:.2f}",
                "Conf": "{:.1f}%",
                "Alpha": "{:.2f}",
                "Winst %": "{:.2f}%"
            }).background_gradient(subset=['Winst %'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Geen signalen gevonden.")
