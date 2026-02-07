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
    """
    Haalt ALLE data op uit Supabase door middel van pagination (in blokjes van 1000).
    """
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        all_rows = []
        start = 0
        batch_size = 1000  # Supabase limiet is vaak 1000 per keer
        
        # We blijven loopen totdat we geen data meer terugkrijgen
        while True:
            response = supabase.table('stock_predictions')\
                .select("*")\
                .range(start, start + batch_size - 1)\
                .execute()
            
            rows = response.data
            
            # Als er geen data meer is, stop de loop
            if not rows:
                break
            
            all_rows.extend(rows)
            
            # Als we minder rijen kregen dan de batch size, zijn we klaar
            if len(rows) < batch_size:
                break
                
            # Voorbereiden voor volgende batch
            start += batch_size

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # 2. Zet ALLE kolomnamen om naar kleine letters (Forceer lowercase)
        df.columns = df.columns.str.lower()
        
        # 3. Check of 'run_date' nu bestaat en converteer
        if 'run_date' in df.columns:
            df['run_date'] = pd.to_datetime(df['run_date'])
            df = df.sort_values('run_date')
        else:
            # Fallback
            if 'date' in df.columns:
                df['run_date'] = pd.to_datetime(df['date'])
                df = df.sort_values('run_date')
            else:
                return pd.DataFrame()
        
        return df

    except Exception as e:
        st.error(f"Fout bij laden data: {e}")
        return pd.DataFrame()

# Laad de data (dit kan nu iets langer duren omdat hij 20x moet verversen op de achtergrond)
data = load_data()

st.title("🎯 Sweet Spot Monitor")

# Debug info (optioneel, haal dit weg als je het storend vindt)
st.caption(f"Totaal aantal rijen ingeladen: {len(data)}")

if data.empty:
    st.warning("Nog geen data gevonden in Supabase.")
    st.stop()

# ==========================================
# 2. REKENFUNCTIE (DE LOGICA)
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    results = []
    
    # Veiligheidscheck
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
# 3. VISUALISATIE (PASTEL GRAFIEK 🎨)
# ==========================================
st.subheader("🔎 Analyse per Aandeel")

unique_tickers = data['ticker'].unique()
ticker = st.selectbox("Selecteer aandeel:", unique_tickers)

if ticker:
    ticker_str = str(ticker)
    subset = data[data['ticker'] == ticker]
    
    fig = go.Figure()

    # Linker Y-As (Alpha - Blauw/Turquoise Pastels)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name='Alpha 2W', 
        line=dict(color='#5D9CEC', width=2)  # Pastel Blauw
    ))
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_4w_norm'],
        name='Alpha 4W', 
        line=dict(color='#4FC1E9', width=2)  # Pastel Turquoise
    ))

    # Rechter Y-As (Confidence - Rood/Oranje Pastels)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_2w'],
        name='Conf 2W', 
        line=dict(color='#ED5565', width=2, dash='dot'), # Pastel Rood/Zalm
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_4w'],
        name='Conf 4W', 
        line=dict(color='#FC6E51', width=2, dash='dot'), # Pastel Oranje
        yaxis='y2'
    ))

    # Layout Update (VEILIGE SYNTAX)
    fig.update_layout(
        title=f"Signaalverloop {ticker_str}",
        xaxis_title="Datum",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0),
        # Y-as Links (Alpha)
        yaxis=dict(
            title=dict(text="Alpha Norm", font=dict(color="#5D9CEC")),
            tickfont=dict(color="#5D9CEC")
        ),
        # Y-as Rechts (Confidence)
        yaxis2=dict(
            title=dict(text="Confidence (%)", font=dict(color="#ED5565")),
            tickfont=dict(color="#ED5565"),
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        # Achtergrond wit maken voor frisse look
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        xaxis=dict(
            showgrid=True, gridcolor='#F0F2F6' # Heel lichtgrijs raster
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
