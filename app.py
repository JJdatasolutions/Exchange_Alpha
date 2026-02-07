import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & FORCE DARK CSS
# ==========================================
st.set_page_config(
    page_title="AlphaTrader Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* 1. FORCEER DONKERE ACHTERGROND VOOR DE HELE APP */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* 2. INPUT VELDEN (ZOEKBALK) FIX - WIT OP WIT VERHELPEN */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] {
        background-color: #262730 !important;
        color: white !important;
        border-color: #444 !important;
    }
    div[data-baseweb="select"] span {
        color: white !important;
    }
    /* De dropdown lijst zelf */
    ul[data-baseweb="menu"] {
        background-color: #262730 !important;
    }
    
    /* 3. METRICS STYLING */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        color: #00CC96 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #A0A0A0 !important;
    }

    /* 4. TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1F2937;
        color: #D1D5DB;
        border: none;
        margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #374151 !important;
        color: #00CC96 !important;
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
# 2. DATA LADEN & VOORBEREIDEN
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
        
        # Datums
        col = 'run_date' if 'run_date' in df.columns else 'date'
        if col in df.columns:
            df['run_date'] = pd.to_datetime(df[col])
            df = df.sort_values(['ticker', 'run_date'])
            
            # BEREKEN DAGELIJKSE PROCENTUELE VERANDERING (Voor de grafiek)
            df['daily_pct'] = df.groupby('ticker')['price'].pct_change() * 100
        
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

data = load_data()

# Header
col_logo, col_info = st.columns([2, 1])
with col_logo:
    st.title("🚀 AlphaTrader Pro")
with col_info:
    if not data.empty:
        last_dt = data['run_date'].max().strftime('%d-%m')
        st.markdown(f"<div style='text-align:right; color:#888; font-size:0.8rem; margin-top:10px;'>Data: {last_dt}</div>", unsafe_allow_html=True)

if data.empty:
    st.warning("Geen data.")
    st.stop()

# ==========================================
# 3. HELPER: DARK TABLE STYLING
# ==========================================
def style_dark_table(df_input):
    """
    Past styling toe op een dataframe zodat hij zwart/grijs is 
    in plaats van standaard wit.
    """
    return df_input.style.format({
        "Prijs": "{:.2f}", 
        "Start": "{:.2f}",
        "Alpha": "{:.2f}", 
        "Conf": "{:.0f}%",
        "Conf%": "{:.0f}%",
        "Winst": "{:.2f}%"
    }).set_properties(**{
        'background-color': '#262730',  # Donkergrijze cellen
        'color': '#E0E0E0',             # Witte tekst
        'border-color': '#444'
    }).background_gradient(subset=['Winst'], cmap='Greens', vmin=0, vmax=20)\
      .background_gradient(subset=['Alpha'], cmap='cool', vmin=1, vmax=5)

# ==========================================
# 4. LOGICA
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

# ==========================================
# 5. TABELLEN (NU DARK MODE)
# ==========================================
st.markdown("### 📡 Market Scanner")
tab1, tab2, tab3 = st.tabs(["🔥 Nu Actief", "⚡ Historie 2W", "🔮 Historie 4W"])

with tab1:
    latest = data['run_date'].max()
    mask_rec = data['run_date'] >= (latest - timedelta(days=3))
    mask_sig = (data['confidence_2w'] > 70) & (data['alpha_2w_norm'] > 1.0)
    active = data[mask_rec & mask_sig].copy()
    
    if not active.empty:
        disp = active[['ticker', 'price', 'alpha_2w_norm', 'confidence_2w']]
        disp.columns = ['Sym', 'Prijs', 'Alpha', 'Conf%']
        
        # Pas de DARK STYLE toe
        st.dataframe(
            style_dark_table(disp).map(lambda x: 'background-color: #262730; color: white'), 
            use_container_width=True, hide_index=True
        )
    else:
        st.caption("Geen breakouts.")

with tab2:
    df2 = get_sweet_spots(data, 21, 'alpha_2w_norm', 'confidence_2w')
    if not df2.empty:
        df2 = df2.sort_values('raw_date', ascending=False)
        cols = ['Sym', 'Datum', 'Start', 'Winst', 'Alpha', 'Conf', 'Stat']
        st.dataframe(
            style_dark_table(df2[cols]), 
            use_container_width=True, hide_index=True, height=300
        )
    else: st.info("Geen data")

with tab3:
    df4 = get_sweet_spots(data, 28, 'alpha_4w_norm', 'confidence_4w')
    if not df4.empty:
        df4 = df4.sort_values('raw_date', ascending=False)
        cols = ['Sym', 'Datum', 'Start', 'Winst', 'Alpha', 'Conf', 'Stat']
        st.dataframe(
            style_dark_table(df4[cols]), 
            use_container_width=True, hide_index=True, height=300
        )
    else: st.info("Geen data")

st.divider()

# ==========================================
# 6. CHART (DYNAMISCHE AS + HOVER %)
# ==========================================
st.markdown("### 🔎 Deep Dive")

tickers = sorted(data['ticker'].unique())
# Selectbox zou nu donker moeten zijn door de CSS bovenaan
selected = st.selectbox("Zoek Aandeel:", tickers, index=0)

if selected:
    subset = data[data['ticker'] == selected].copy()
    last = subset.iloc[-1]
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prijs", f"{last['price']:.2f}")
    k2.metric("Alpha 2W", f"{last['alpha_2w_norm']:.2f}")
    k3.metric("Conf 2W", f"{last['confidence_2w']:.0f}%")
    k4.metric("Conf 4W", f"{last['confidence_4w']:.0f}%")

    # --- BEREKEN Y-AS RANGE ---
    # We nemen de min en max van de prijs en voegen 2% margin toe
    min_price = subset['price'].min()
    max_price = subset['price'].max()
    y_padding = (max_price - min_price) * 0.1 # 10% padding
    if y_padding == 0: y_padding = 1
    
    y_range = [min_price - y_padding, max_price + y_padding]

    # --- MAAK CHART ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 1. PRIJS LIJN (Met Hover Info)
    # We maken een aangepaste hover tekst
    subset['hover_text'] = subset.apply(
        lambda x: f"Datum: {x['run_date'].strftime('%d-%m')}<br>Prijs: {x['price']:.2f}<br>Dagwinst: {'🟢' if x['daily_pct'] >= 0 else '🔴'} {x['daily_pct']:.2f}%", 
        axis=1
    )

    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['price'],
        name="Prijs",
        line=dict(color='#00CC96', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)',
        text=subset['hover_text'], # <--- HIER ZIT DE MAGIC
        hoverinfo='text'           # Forceer onze eigen tekst
    ), row=1, col=1)

    # 2. SIGNALEN
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['alpha_2w_norm'], name="Alpha 2W", line=dict(color='#26C6DA', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['alpha_4w_norm'], name="Alpha 4W", line=dict(color='#AB47BC', width=2)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['confidence_2w'], name="Conf 2W", line=dict(color='#FF7043', width=1, dash='dot')), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=subset['run_date'], y=subset['confidence_4w'], name="Conf 4W", line=dict(color='#FFA726', width=1, dash='dot')), row=2, col=1, secondary_y=True)

    # LAYOUT
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.05, x=0, font=dict(color="#AAA")),
        hovermode="x unified"
    )
    
    # ASSEN CONFIGURATIE (DYNAMISCH!)
    # Hier zetten we de berekende range
    fig.update_yaxes(
        range=y_range, # <--- DIT FIKST HET "PLATTE LIJN" PROBLEEM
        title="Prijs", row=1, col=1, 
        showgrid=True, gridcolor='#333', color='#BBB'
    )
    
    fig.update_yaxes(title="Alpha", row=2, col=1, showgrid=True, gridcolor='#333', color='#26C6DA', secondary_y=False)
    fig.update_yaxes(title="Conf", row=2, col=1, showgrid=False, color='#FF7043', secondary_y=True, range=[0,105])
    
    fig.update_xaxes(showgrid=True, gridcolor='#333', color='#BBB')

    st.plotly_chart(fig, use_container_width=True)
