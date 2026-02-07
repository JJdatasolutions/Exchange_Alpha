import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & DESIGN (STEALTH DARK)
# ==========================================
st.set_page_config(
    page_title="AlphaTrader Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Main Dark Theme */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Input Fields Styling */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] {
        background-color: #1F2937 !important;
        color: white !important;
        border: 1px solid #374151 !important;
    }
    div[data-baseweb="select"] span {
        color: white !important;
    }
    ul[data-baseweb="menu"] {
        background-color: #1F2937 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        color: #00CC96 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9CA3AF !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        color: #9CA3AF;
        border-radius: 4px;
        border: none;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #374151 !important;
        color: #00CC96 !important;
        font-weight: bold;
    }
    
    /* Plotly Chart Background Fix */
    .js-plotly-plot .plotly .main-svg {
        background-color: rgba(0,0,0,0) !important;
    }

    /* --- TABEL HEADER FIX --- */
    /* Probeert de headers donker te maken via CSS */
    [data-testid="stDataFrame"] th {
        background-color: #1F2937 !important;
        color: #E0E0E0 !important;
        border-bottom: 1px solid #444 !important;
    }
    /* Fallback voor oudere browsers/streamlit versies */
    thead tr th {
        background-color: #1F2937 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except FileNotFoundError:
    st.error("Secrets missing.")
    st.stop()

# ==========================================
# 2. DATA
# ==========================================
@st.cache_data(ttl=600)
def load_data():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        all_rows = []
        start = 0
        batch_size = 1000
        while True:
            response = supabase.table('stock_predictions').select("*").range(start, start + batch_size - 1).execute()
            rows = response.data
            if not rows: break
            all_rows.extend(rows)
            if len(rows) < batch_size: break
            start += batch_size

        if not all_rows: return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df.columns = df.columns.str.lower()
        
        col = 'run_date' if 'run_date' in df.columns else 'date'
        if col in df.columns:
            df['run_date'] = pd.to_datetime(df[col])
            df = df.sort_values(['ticker', 'run_date'])
            df['daily_pct'] = df.groupby('ticker')['price'].pct_change() * 100
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

data = load_data()

# Header
c1, c2 = st.columns([3, 1])
with c1: st.title("🚀 AlphaTrader Pro")
with c2: 
    if not data.empty:
        st.markdown(f"<div style='text-align:right; color:#666; font-size:0.8em; padding-top:10px;'>Update: {data['run_date'].max().strftime('%d-%m')}</div>", unsafe_allow_html=True)

if data.empty: st.stop()

# ==========================================
# 3. LOGICA & VISUALISATIES
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
            status = "🏁" if days >= lookahead_days else f"⏳{days}d"
        else:
            max_px = start_px
            status = "🆕"
            
        pct = ((max_px - start_px) / start_px) * 100
        results.append({
            "Sym": ticker,
            "Datum": start_dt.strftime('%d-%m'),
            "Start": start_px,
            "Winst": pct,
            "Conf": row[conf_col],
            "Alpha": row[alpha_col],
            "Stat": status,
            "raw_date": start_dt
        })
    return pd.DataFrame(results)

def apply_table_style(df_input):
    # CSS properties voor de headers specifiek in Pandas
    headers = {
        'selector': 'th',
        'props': [('background-color', '#1F2937'), ('color', 'white'), ('border-bottom', '2px solid #374151')]
    }
    
    return df_input.style.format({
        "Prijs": "{:.2f}", "Start": "{:.2f}",
        "Alpha": "{:.2f}", "Conf": "{:.0f}%", "Conf%": "{:.0f}%",
        "Winst": "{:.1f}%"
    }).set_properties(**{
        'background-color': '#1F2937', 
        'color': '#E5E7EB', 
        'border-color': '#374151'
    })\
    .set_table_styles([headers])\
    .background_gradient(subset=['Winst'], cmap='Greens', vmin=0, vmax=25)
    # GEEN KLEUR MEER OP ALPHA

# --- STATISCHE BOXPLOT ---
def plot_distribution(df_in):
    if df_in.empty: return
    
    avg_gain = df_in['Winst'].mean()
    median_gain = df_in['Winst'].median()
    max_gain = df_in['Winst'].max()
    
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Gem. Winst", f"{avg_gain:.2f}%")
    k2.metric("Mediaan", f"{median_gain:.2f}%")
    k3.metric("Max Uitschieter", f"{max_gain:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Box(
        x=df_in['Winst'], name='Spreiding', marker_color='#00CC96',
        boxmean=True, orientation='h', boxpoints='outliers', line=dict(width=1.5),
        hoverinfo='skip'
    ))

    fig.update_layout(
        template="plotly_dark", height=120,             
        margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(title="Winst %", showgrid=True, gridcolor='#333'),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, hovermode=False, dragmode=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==========================================
# 4. VIEW: TABS & TABLES
# ==========================================
st.markdown("### 📡 Market Scanner")
tab_act, tab_2w, tab_4w = st.tabs(["🔥 Nu Actief", "⚡ Historie 2W", "🔮 Historie 4W"])

with tab_act:
    latest = data['run_date'].max()
    mask_rec = data['run_date'] >= (latest - timedelta(days=3))
    mask_sig = (data['confidence_2w'] > 70) & (data['alpha_2w_norm'] > 1.0)
    active = data[mask_rec & mask_sig].copy()
    
    if not active.empty:
        st.info(f"Signalen van {latest.strftime('%d-%m')}")
        disp = active[['ticker', 'price', 'alpha_2w_norm', 'confidence_2w']]
        disp.columns = ['Sym', 'Prijs', 'Alpha', 'Conf%']
        
        st.dataframe(apply_table_style(disp), use_container_width=True, hide_index=True)
    else:
        st.caption("Geen signalen.")

with tab_2w:
    df2 = get_sweet_spots(data, 21, 'alpha_2w_norm', 'confidence_2w')
    if not df2.empty:
        plot_distribution(df2)
        df2 = df2.sort_values('raw_date', ascending=False)
        cols = ['Sym', 'Datum', 'Start', 'Alpha', 'Conf', 'Winst', 'Stat']
        st.dataframe(apply_table_style(df2[cols]), use_container_width=True, hide_index=True, height=300)
    else: st.info("Geen data")

with tab_4w:
    df4 = get_sweet_spots(data, 28, 'alpha_4w_norm', 'confidence_4w')
    if not df4.empty:
        plot_distribution(df4)
        df4 = df4.sort_values('raw_date', ascending=False)
        cols = ['Sym', 'Datum', 'Start', 'Alpha', 'Conf', 'Winst', 'Stat']
        st.dataframe(apply_table_style(df4[cols]), use_container_width=True, hide_index=True, height=300)
    else: st.info("Geen data")

st.divider()

# ==========================================
# 5. CHART & ANALYSE
# ==========================================
st.markdown("### 🔎 Deep Dive")

tickers = sorted(data['ticker'].unique())
selected = st.selectbox("Zoek Aandeel:", tickers, index=0)

if selected:
    subset = data[data['ticker'] == selected].copy()
    last = subset.iloc[-1]
    
    # 5 COLUMNS VOOR DE KPI's
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Prijs", f"{last['price']:.2f}")
    k2.metric("Alpha 2W", f"{last['alpha_2w_norm']:.2f}")
    k3.metric("Alpha 4W", f"{last['alpha_4w_norm']:.2f}") # <--- NIEUW: ALPHA 4W
    k4.metric("Conf 2W", f"{last['confidence_2w']:.0f}%")
    k5.metric("Conf 4W", f"{last['confidence_4w']:.0f}%")

    min_px, max_px = subset['price'].min(), subset['price'].max()
    pad = (max_px - min_px) * 0.05
    if pad == 0: pad = 0.5
    y_range = [min_px - pad, max_px + pad]

    subset['hov'] = subset.apply(lambda x: f"Datum: {x['run_date'].strftime('%d-%m')}<br>Prijs: {x['price']:.2f}<br>Dag: {'🟢' if x['daily_pct']>=0 else '🔴'} {x['daily_pct']:.2f}%", axis=1)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.4], specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. PRIJS
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['price'], name="Prijs",
        line=dict(color='#00CC96', width=2), fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)',
        text=subset['hov'], hoverinfo='text'
    ), row=1, col=1)

    # 2. SIGNALEN
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['alpha_2w_norm'], name="Alpha 2W", line=dict(color='#26C6DA', width=2)), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['alpha_4w_norm'], name="Alpha 4W", line=dict(color='#FF00FF', width=2)), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['confidence_2w'], name="Conf 2W", line=dict(color='#F87171', width=1, dash='dot')), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['confidence_4w'], name="Conf 4W", line=dict(color='#FBBF24', width=1, dash='dot')), row=2, col=1, secondary_y=True)

    fig.update_layout(
        template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified", 
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color="white"), bgcolor="rgba(0,0,0,0)")
    )
    
    fig.update_yaxes(range=y_range, row=1, col=1, gridcolor='#333', color='#AAA')
    fig.update_yaxes(title="Alpha", row=2, col=1, gridcolor='#333', color='#26C6DA', secondary_y=False)
    fig.update_yaxes(title="Conf", row=2, col=1, showgrid=False, color='#F87171', secondary_y=True, range=[0,105])

    st.plotly_chart(fig, use_container_width=True)
