import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & DESIGN SYSTEM
# ==========================================
st.set_page_config(
    page_title="AlphaTrader Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS voor betere integratie van tabellen in Dark Mode
st.markdown("""
    <style>
    /* Algemene achtergrond fix */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Headers en tekst kleur */
    h1, h2, h3, p, div, span {
        color: #FAFAFA !important;
    }
    
    /* Metrics stylen (Grote getallen) */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        color: #00CC96 !important; /* Emerald */
    }
    [data-testid="stMetricLabel"] {
        color: #A0A0A0 !important; /* Grijs */
    }
    
    /* Tabs: Donkere achtergrond en accenten */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1F2937; /* Donkergrijs blokje */
        border-radius: 4px;
        color: #D1D5DB;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #374151 !important; /* Iets lichter bij selectie */
        color: #00CC96 !important;
        border-bottom: 2px solid #00CC96;
    }
    
    /* DataFrame aanpassingen om het witte vlak te verminderen */
    [data-testid="stDataFrame"] {
        background-color: #1F2937;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Secrets ophalen
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
        
        # Datum check
        col = 'run_date' if 'run_date' in df.columns else 'date'
        if col in df.columns:
            df['run_date'] = pd.to_datetime(df[col])
            df = df.sort_values('run_date')
        
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

data = load_data()

# Header
col_logo, col_info = st.columns([3, 1])
with col_logo:
    st.title("🚀 AlphaTrader Pro")
with col_info:
    if not data.empty:
        last_dt = data['run_date'].max().strftime('%d-%m')
        st.markdown(f"<div style='text-align:right; color:#888;'>Data: {last_dt}</div>", unsafe_allow_html=True)

if data.empty:
    st.warning("Geen data.")
    st.stop()

# ==========================================
# 3. REKENLOGICA (MET ALLE KOLOMMEN)
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    if alpha_col not in df.columns: return pd.DataFrame()

    # Filter
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
            "Conf": row[conf_col],       # TERUG TOEGEVOEGD
            "Alpha": row[alpha_col],     # TERUG TOEGEVOEGD
            "Stat": status,
            "raw_date": start_dt
        })
    return pd.DataFrame(results)

# ==========================================
# 4. VIEW: TABELLEN
# ==========================================
st.markdown("### 📡 Market Scanner")

tab_active, tab_hist_2w, tab_hist_4w = st.tabs(["🔥 Nu Actief", "⚡ Historie 2W", "🔮 Historie 4W"])

# --- TAB 1: NU ACTIEF ---
with tab_active:
    latest_date = data['run_date'].max()
    # Pak laatste 3 dagen (voor weekend correctie)
    mask_recent = data['run_date'] >= (latest_date - timedelta(days=3))
    mask_signal = (data['confidence_2w'] > 70) & (data['alpha_2w_norm'] > 1.0)
    
    active = data[mask_recent & mask_signal].copy()
    
    if not active.empty:
        st.info(f"Signalen van {latest_date.strftime('%d-%m')}")
        disp = active[['ticker', 'price', 'alpha_2w_norm', 'confidence_2w']]
        disp.columns = ['Sym', 'Prijs', 'Alpha', 'Conf%']
        
        st.dataframe(
            disp.style.format({
                "Prijs": "{:.2f}", "Alpha": "{:.2f}", "Conf%": "{:.0f}%"
            }).background_gradient(subset=['Alpha'], cmap='cool'),
            use_container_width=True, hide_index=True
        )
    else:
        st.caption("Geen breakouts op de laatste handelsdag.")

# --- TAB 2 & 3: HISTORIE ---
def show_history_table(tab, df_res):
    with tab:
        if not df_res.empty:
            avg = df_res['Winst'].mean()
            col_kpi, _ = st.columns([2,3])
            col_kpi.metric("Gem. Potentieel", f"{avg:.2f}%")
            
            df_res = df_res.sort_values('raw_date', ascending=False)
            
            # Selecteer kolommen inclusief Conf en Alpha
            cols = ['Sym', 'Datum', 'Start', 'Alpha', 'Conf', 'Winst', 'Stat']
            
            st.dataframe(
                df_res[cols].style.format({
                    "Start": "{:.2f}", 
                    "Alpha": "{:.2f}", 
                    "Conf": "{:.0f}%", 
                    "Winst": "{:.1f}%"
                }).background_gradient(subset=['Winst'], cmap='Greens'),
                use_container_width=True, hide_index=True, height=350
            )
        else:
            st.info("Geen historie.")

df2 = get_sweet_spots(data, 21, 'alpha_2w_norm', 'confidence_2w')
df4 = get_sweet_spots(data, 28, 'alpha_4w_norm', 'confidence_4w')

show_history_table(tab_hist_2w, df2)
show_history_table(tab_hist_4w, df4)

st.divider()

# ==========================================
# 5. DIEPGAANDE ANALYSE (CHART DESIGN)
# ==========================================
st.markdown("### 🔎 Deep Dive")

tickers = sorted(data['ticker'].unique())
selected = st.selectbox("Zoek Aandeel:", tickers, index=0)

if selected:
    subset = data[data['ticker'] == selected]
    last = subset.iloc[-1]
    
    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prijs", f"{last['price']:.2f}")
    k2.metric("Alpha 2W", f"{last['alpha_2w_norm']:.2f}")
    k3.metric("Conf 2W", f"{last['confidence_2w']:.0f}%")
    k4.metric("Conf 4W", f"{last['confidence_4w']:.0f}%")

    # --- CHART: DARK MODE & LEESBAARHEID ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.55, 0.45],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. PRIJS (Boven)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['price'],
        name="Prijs",
        line=dict(color='#00CC96', width=2), # Emerald Green
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)' # Zeer lichte groene gloed
    ), row=1, col=1)

    # 2. SIGNALEN (Onder) - 4 LIJNEN
    
    # Alpha (Linker as)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name="Alpha 2W",
        line=dict(color='#26C6DA', width=2) # Cyaan
    ), row=2, col=1, secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_4w_norm'],
        name="Alpha 4W",
        line=dict(color='#AB47BC', width=2) # Paars
    ), row=2, col=1, secondary_y=False)

    # Confidence (Rechter as)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_2w'],
        name="Conf 2W",
        line=dict(color='#FF7043', width=1, dash='dot') # Coral
    ), row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_4w'],
        name="Conf 4W",
        line=dict(color='#FFA726', width=1, dash='dot') # Oranje
    ), row=2, col=1, secondary_y=True)

    # LAYOUT STYLING VOOR DARK MODE
    fig.update_layout(
        template="plotly_dark", # Basis dark theme
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        
        # Achtergrond transparant maken zodat het blendt met de app
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        
        # Legende styling (Wit/Grijs)
        legend=dict(
            orientation="h", y=1.1, x=0,
            font=dict(color="#E0E0E0")
        ),
        hovermode="x unified"
    )
    
    # Assen Styling (Zichtbaar maken op donkere achtergrond)
    grid_col = "#333333" # Donkergrijs raster
    text_col = "#B0B0B0" # Lichtgrijze tekst
    
    # Bovenste grafiek (Prijs)
    fig.update_yaxes(title="Prijs", row=1, col=1, 
                     showgrid=True, gridcolor=grid_col, color=text_col)
    
    # Onderste grafiek (Linker as = Alpha)
    fig.update_yaxes(title="Alpha", row=2, col=1, secondary_y=False,
                     showgrid=True, gridcolor=grid_col, color="#26C6DA") # Cyaan tekst
    
    # Onderste grafiek (Rechter as = Conf)
    fig.update_yaxes(title="Conf %", row=2, col=1, secondary_y=True,
                     showgrid=False, range=[0,105], color="#FF7043") # Coral tekst
    
    # X-as
    fig.update_xaxes(showgrid=True, gridcolor=grid_col, color=text_col)

    st.plotly_chart(fig, use_container_width=True)
