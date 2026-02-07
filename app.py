import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & SETUP
# ==========================================
st.set_page_config(page_title="Stock Sweet Spots", layout="wide")

# Secrets ophalen
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except FileNotFoundError:
    st.error("Geen secrets gevonden. Stel SUPABASE_URL en SUPABASE_KEY in.")
    st.stop()

@st.cache_data(ttl=600)
def load_data():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 1. Haal data op
        response = supabase.table('stock_predictions').select("*").execute()
        
        if not response.data:
            return pd.DataFrame()

        df = pd.DataFrame(response.data)

        # 2. Zet ALLE kolomnamen om naar kleine letters (Forceer lowercase)
        df.columns = df.columns.str.lower()
        
        # 3. Check of 'run_date' nu bestaat
        if 'run_date' in df.columns:
            df['run_date'] = pd.to_datetime(df['run_date'])
            df = df.sort_values('run_date')
        else:
            # Fallback: probeer 'date' of 'created_at' als run_date mist
            if 'date' in df.columns:
                df['run_date'] = pd.to_datetime(df['date'])
                df = df.sort_values('run_date')
            else:
                st.error("Kolom 'run_date' ontbreekt in de database.")
                return pd.DataFrame()
        
        return df

    except Exception as e:
        st.error(f"Fout bij laden data: {e}")
        return pd.DataFrame()

data = load_data()

st.title("🎯 Sweet Spot Monitor")

if data.empty:
    st.warning("Nog geen data gevonden in Supabase. Wacht op de eerste run.")
    st.stop()

# ==========================================
# 2. REKENFUNCTIE (DE LOGICA)
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    results = []
    
    # Veiligheidscheck: bestaan de kolommen?
    if alpha_col not in df.columns or conf_col not in df.columns:
        return pd.DataFrame()

    signals = df[(df[conf_col] > 70) & (df[alpha_col] > 1)].copy()
    
    for idx, row in signals.iterrows():
        ticker = row['ticker']
        start_date = row['run_date']
        start_price = row['price']
        
        end_date = start_date + timedelta(days=lookahead_days)
        
        future_mask = (
            (df['ticker'] == ticker) & 
            (df['run_date'] > start_date) & 
            (df['run_date'] <= end_date)
        )
        future_data = df[future_mask]
        
        if not future_data.empty:
            max_price_seen = future_data['price'].max()
            days_data_available = (future_data['run_date'].max() - start_date).days
        else:
            max_price_seen = start_price
            days_data_available = 0
            
        pct_diff = ((max_price_seen - start_price) / start_price) * 100
        
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

unique_tickers = data['ticker'].unique()
ticker = st.selectbox("Selecteer aandeel:", unique_tickers)

if ticker:
    # Forceer ticker naar string om fouten te voorkomen
    ticker_str = str(ticker)
    subset = data[data['ticker'] == ticker]
    
    fig = go.Figure()

    # Linker Y-As
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name='Alpha 2W', line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_4w_norm'],
        name='Alpha 4W', line=dict(color='cyan', width=2)
    ))

    # Rechter Y-As
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

    # Layout Update (VEILIGE MODERNE SYNTAX)
    fig.update_layout(
        title=f"Signaalverloop {ticker_str}",
        xaxis_title="Datum",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0),
        # Y-as Links (Alpha)
        yaxis=dict(
            title=dict(text="Alpha Norm", font=dict(color="blue")),
            tickfont=dict(color="blue")
        ),
        # Y-as Rechts (Confidence)
        yaxis2=dict(
            title=dict(text="Confidence (%)", font=dict(color="red")),
            tickfont=dict(color="red"),
            overlaying="y",
            side="right",
            range=[0, 100]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================
# 4. DE TABELLEN
# ==========================================

col1, col2 = st.columns(2)

# --- TABEL 1 ---
with col1:
    st.markdown("### ⚡ Sweet Spot: 2 Weken")
    df_2w = get_sweet_spots(data, lookahead_days=21, alpha_col='alpha_2w_norm', conf_col='confidence_2w')
    
    if not df_2w.empty:
        avg_gain = df_2w['Winst %'].mean()
        st.metric("Gemiddelde Max Winst", f"{avg_gain:.2f}%")
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
        st.info("Geen signalen gevonden.")

# --- TABEL 2 ---
with col2:
    st.markdown("### 🔮 Sweet Spot: 4 Weken")
    df_4w = get_sweet_spots(data, lookahead_days=28, alpha_col='alpha_4w_norm', conf_col='confidence_4w')
    
    if not df_4w.empty:
        avg_gain = df_4w['Winst %'].mean()
        st.metric("Gemiddelde Max Winst", f"{avg_gain:.2f}%")
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
