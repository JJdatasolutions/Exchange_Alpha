import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & DARK MODE STYLING
# ==========================================
st.set_page_config(
    page_title="AlphaTrader Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS voor een strakke Dark Mode look
st.markdown("""
    <style>
    /* Algemene achtergrond donker maken (als Streamlit niet in dark mode staat) */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Metrics (KPI's) strakker maken */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #00CC96 !important; /* Cyber Green */
    }
    [data-testid="stMetricLabel"] {
        color: #979797 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E1E1E;
        border-radius: 5px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00CC96 !important;
        color: black !important;
    }
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
# 2. DATA LADEN
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
        col = 'run_date' if 'run_date' in df.columns else 'date'
        if col in df.columns:
            df['run_date'] = pd.to_datetime(df[col])
            df = df.sort_values('run_date')
        
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

data = load_data()

# Header Area
if not data.empty:
    last_date = data['run_date'].max()
    last_date_str = last_date.strftime('%d %B')
else:
    last_date_str = "..."

col_logo, col_info = st.columns([2, 1])
with col_logo:
    st.title("🚀 AlphaTrader Pro")
with col_info:
    st.markdown(f"<div style='text-align: right; color: gray;'>Laatste Data: {last_date_str}</div>", unsafe_allow_html=True)

if data.empty:
    st.warning("Geen data beschikbaar.")
    st.stop()

# ==========================================
# 3. REKENLOGICA (HISTORIE & ACTIEF)
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    if alpha_col not in df.columns: return pd.DataFrame()

    # Filter op sterke signalen
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
            status = "🏁 Klaar" if days >= lookahead_days else f"⏳ {days}d"
        else:
            max_px = start_px
            status = "🆕 Nieuw"
            
        pct = ((max_px - start_px) / start_px) * 100
        results.append({
            "Sym": ticker,
            "Datum": start_dt.strftime('%d-%m'),
            "Start": start_px,
            "Winst": pct,
            "Conf": row[conf_col],
            "Alpha": row[alpha_col],
            "Status": status,
            "raw_date": start_dt # Voor sorteren
        })
    return pd.DataFrame(results)

# ==========================================
# 4. VIEW: HOT PICKS & HISTORIE
# ==========================================
st.markdown("### 📡 Market Scanner")

# Tabs
tab_active, tab_hist_2w, tab_hist_4w = st.tabs(["🔥 Actieve Signalen", "⚡ Historie (2W)", "🔮 Historie (4W)"])

# --- TAB 1: ACTIEVE SIGNALEN (DE BREAKOUT KANSHEBBERS) ---
with tab_active:
    # We kijken naar signalen van de LAATSTE datumbeschikbaar in de dataset
    latest_date = data['run_date'].max()
    # Filter: Datum is recent (laatste 3 dagen voor weekend effect) EN goede signalen
    mask_recent = data['run_date'] >= (latest_date - timedelta(days=3))
    mask_signal = (data['confidence_2w'] > 70) & (data['alpha_2w_norm'] > 1.0)
    
    active_df = data[mask_recent & mask_signal].copy()
    
    if not active_df.empty:
        st.info(f"Deze aandelen gaven een koopsignaal op {latest_date.strftime('%d-%m')}.")
        # Toon tabel
        display_cols = active_df[['ticker', 'price', 'alpha_2w_norm', 'confidence_2w']]
        display_cols.columns = ['Symbool', 'Prijs', 'Alpha', 'Conf %']
        
        st.dataframe(
            display_cols.style.format({
                "Prijs": "{:.2f}", "Alpha": "{:.2f}", "Conf %": "{:.0f}%"
            }).background_gradient(subset=['Alpha'], cmap='cool'), # Cool = blauw/paars
            use_container_width=True, hide_index=True
        )
    else:
        st.write("Geen actieve breakouts gevonden op de laatste datum.")

# --- TAB 2 & 3: HISTORISCHE PERFORMANCE ---
def show_history_tab(tab, df_res):
    with tab:
        if not df_res.empty:
            # 2. KPI: GEMIDDELDE WINST (TERUG VAN WEGGEWEEST)
            avg_gain = df_res['Winst'].mean()
            col_kpi, _ = st.columns([1,3])
            col_kpi.metric("Gem. Max Rendement", f"{avg_gain:.2f}%")
            
            # Sorteer op datum (nieuwste eerst)
            df_res = df_res.sort_values('raw_date', ascending=False)
            
            st.dataframe(
                df_res[['Sym', 'Datum', 'Start', 'Winst', 'Status']].style.format({
                    "Start": "{:.2f}", "Winst": "{:.1f}%"
                }).background_gradient(subset=['Winst'], cmap='Greens'),
                use_container_width=True, hide_index=True, height=300
            )
        else:
            st.info("Geen historische signalen.")

df_2w = get_sweet_spots(data, 21, 'alpha_2w_norm', 'confidence_2w')
df_4w = get_sweet_spots(data, 28, 'alpha_4w_norm', 'confidence_4w')

show_history_tab(tab_hist_2w, df_2w)
show_history_tab(tab_hist_4w, df_4w)

st.divider()

# ==========================================
# 5. DIEPGAANDE ANALYSE (DARK CHART)
# ==========================================
st.markdown("### 🔎 Deep Dive")

tickers = sorted(data['ticker'].unique())
selected = st.selectbox("Selecteer Aandeel:", tickers, index=0)

if selected:
    subset = data[data['ticker'] == selected]
    last = subset.iloc[-1]
    
    # KPI Cards boven grafiek
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prijs", f"{last['price']:.2f}")
    k2.metric("Alpha 2W", f"{last['alpha_2w_norm']:.2f}", delta_color="off")
    k3.metric("Conf 2W", f"{last['confidence_2w']:.0f}%", delta_color="off")
    k4.metric("Conf 4W", f"{last['confidence_4w']:.0f}%", delta_color="off")

    # --- DARK MODE CHART MET 4 LIJNEN ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. Prijs (Boven) - Area Chart
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['price'],
        name="Prijs",
        line=dict(color='#00CC96', width=2), # Cyber Green
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)'
    ), row=1, col=1)

    # 2. Signalen (Onder) - 4 LIJNEN
    
    # Alpha lijnen (Linker As)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name="Alpha 2W",
        line=dict(color='#00F2EA', width=2) # Neon Cyaan
    ), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_4w_norm'],
        name="Alpha 4W",
        line=dict(color='#B300FF', width=2) # Neon Paars
    ), row=2, col=1, secondary_y=False)

    # Confidence lijnen (Rechter As)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_2w'],
        name="Conf 2W",
        line=dict(color='#FF0055', width=1, dash='dot') # Neon Rood
    ), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_4w'],
        name="Conf 4W",
        line=dict(color='#FFAA00', width=1, dash='dot') # Neon Oranje
    ), row=2, col=1, secondary_y=True)

    # Layout Styling (DARK MODE)
    fig.update_layout(
        template="plotly_dark", # <--- DIT DOET HET ZWARE WERK
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            y=1.1,
            x=0
        ),
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', # Transparant zodat CSS erdoor komt
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Assen configureren
    fig.update_yaxes(title="Prijs", row=1, col=1, showgrid=True, gridcolor='#333')
    fig.update_yaxes(title="Alpha", row=2, col=1, showgrid=True, gridcolor='#333', secondary_y=False)
    fig.update_yaxes(title="Conf %", row=2, col=1, showgrid=False, secondary_y=True, range=[0,100])

    st.plotly_chart(fig, use_container_width=True)
