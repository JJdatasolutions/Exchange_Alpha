import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & CSS (MOBILE FIRST)
# ==========================================
st.set_page_config(
    page_title="AlphaTrader", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS voor mobiele optimalisatie
st.markdown("""
    <style>
    /* Minder witruimte bovenaan */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }
    /* Knoppen en tabs iets groter voor touch */
    button {
        min-height: 45px !important;
    }
    /* Metriek labels iets kleiner, waarden groter */
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    </style>
""", unsafe_allow_html=True)

# Secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except FileNotFoundError:
    st.error("Geen secrets gevonden.")
    st.stop()

# ==========================================
# 2. DATA LADEN (ROBUUST)
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        all_rows = []
        start = 0
        batch_size = 1000
        
        while True:
            response = supabase.table('stock_predictions')\
                .select("*")\
                .range(start, start + batch_size - 1)\
                .execute()
            rows = response.data
            if not rows: break
            all_rows.extend(rows)
            if len(rows) < batch_size: break
            start += batch_size

        if not all_rows: return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df.columns = df.columns.str.lower()
        
        # Datum correctie
        date_col = 'run_date' if 'run_date' in df.columns else 'date'
        if date_col in df.columns:
            df['run_date'] = pd.to_datetime(df[date_col])
            df = df.sort_values('run_date')
        
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

data = load_data()

# Header
col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("🚀 AlphaTrader")
with col_stat:
    if not data.empty:
        st.caption(f"Upd: {data['run_date'].max().strftime('%d-%m')}")

if data.empty:
    st.warning("Geen data...")
    st.stop()

# ==========================================
# 3. REKENLOGICA
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    if alpha_col not in df.columns: return pd.DataFrame()

    signals = df[(df[conf_col] > 70) & (df[alpha_col] > 1)].copy()
    results = []
    
    for _, row in signals.iterrows():
        ticker = row['ticker']
        start_dt = row['run_date']
        start_px = row['price']
        end_dt = start_dt + timedelta(days=lookahead_days)
        
        future = df[(df['ticker'] == ticker) & (df['run_date'] > start_dt) & (df['run_date'] <= end_dt)]
        
        if not future.empty:
            max_px = future['price'].max()
            days = (future['run_date'].max() - start_dt).days
            status = "🏁" if days >= lookahead_days else f"⏳ {days}d"
        else:
            max_px = start_px
            status = "🆕"
            
        pct = ((max_px - start_px) / start_px) * 100
        results.append({
            "Sym": ticker,
            "Datum": start_dt.strftime('%d-%m'),
            "Start": start_px,
            "Winst": pct,
            "Stat": status
        })
    return pd.DataFrame(results)

# ==========================================
# 4. BOVENKANT: TABELLEN
# ==========================================
st.markdown("### 🏆 Top Picks")
tab1, tab2 = st.tabs(["⚡ 2 Weken", "🔮 4 Weken"])

with tab1:
    df2 = get_sweet_spots(data, 21, 'alpha_2w_norm', 'confidence_2w')
    if not df2.empty:
        st.dataframe(
            df2[['Sym', 'Datum', 'Start', 'Winst', 'Stat']].style.format({
                "Start": "{:.2f}", "Winst": "{:.1f}%"
            }).background_gradient(subset=['Winst'], cmap='Greens'),
            use_container_width=True, hide_index=True, height=250
        )
    else: st.info("Geen signalen")

with tab2:
    df4 = get_sweet_spots(data, 28, 'alpha_4w_norm', 'confidence_4w')
    if not df4.empty:
        st.dataframe(
            df4[['Sym', 'Datum', 'Start', 'Winst', 'Stat']].style.format({
                "Start": "{:.2f}", "Winst": "{:.1f}%"
            }).background_gradient(subset=['Winst'], cmap='Greens'),
            use_container_width=True, hide_index=True, height=250
        )
    else: st.info("Geen signalen")

st.divider()

# ==========================================
# 5. ONDERKANT: ANALYSE & CHARTS
# ==========================================
st.markdown("### 🔎 Analyse")

# Zoekbalk
tickers = sorted(data['ticker'].unique())
selected = st.selectbox("Zoek aandeel:", tickers, index=0)

if selected:
    subset = data[data['ticker'] == selected]
    last = subset.iloc[-1]
    
    # KPI Cards
    k1, k2, k3 = st.columns(3)
    k1.metric("Prijs", f"{last['price']:.2f}")
    k2.metric("Alpha", f"{last['alpha_2w_norm']:.2f}")
    k3.metric("Conf", f"{last['confidence_2w']:.0f}%")

    # --- DE CORRECTE SUBPLOT ---
    # Hier fixen we de error door 'specs' mee te geven
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4], # Prijs grafiek iets groter dan signaal grafiek
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]] # <--- DE FIX!
    )

    # 1. Prijs Grafiek (Boven)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['price'],
        name="Prijs",
        line=dict(color='#10B981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ), row=1, col=1)

    # 2. Signaal Grafiek (Onder - Linker As)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name="Alpha",
        line=dict(color='#5D9CEC', width=2)
    ), row=2, col=1, secondary_y=False)

    # 3. Confidence (Onder - Rechter As)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_2w'],
        name="Conf",
        line=dict(color='#ED5565', width=2, dash='dot')
    ), row=2, col=1, secondary_y=True)

    # Layout Styling
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode="x unified"
    )
    
    # Assen configureren
    fig.update_yaxes(title="Prijs", row=1, col=1, showgrid=True, gridcolor='#F0F2F6')
    fig.update_yaxes(title="Alpha", row=2, col=1, showgrid=True, gridcolor='#F0F2F6', secondary_y=False)
    fig.update_yaxes(title="Conf %", row=2, col=1, showgrid=False, secondary_y=True, range=[0,100], tickfont=dict(color='#ED5565'))

    st.plotly_chart(fig, use_container_width=True)
