import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import timedelta

# ==========================================
# 1. CONFIGURATIE & CSS (STYLING)
# ==========================================
st.set_page_config(
    page_title="AlphaTrader Mobile", 
    layout="wide",
    initial_sidebar_state="collapsed" # Meer ruimte op mobiel
)

# Custom CSS voor een 'App-like' ervaring
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0rem;
    }
    h3 {
        font-size: 1.2rem !important;
        margin-top: 1rem;
    }
    /* Maak metrics mooier op mobiel */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    /* Tabs iets groter maken voor touch */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        padding: 10px;
        flex: 1; /* Tabs vullen de breedte */
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

@st.cache_data(ttl=600)
def load_data():
    """ Haalt ALLE data op (met pagination loop) """
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
        
        if 'run_date' in df.columns:
            df['run_date'] = pd.to_datetime(df['run_date'])
            df = df.sort_values('run_date')
        elif 'date' in df.columns:
            df['run_date'] = pd.to_datetime(df['date'])
            df = df.sort_values('run_date')
        else:
            return pd.DataFrame()
        
        return df
    except Exception as e:
        st.error(f"Fout: {e}")
        return pd.DataFrame()

data = load_data()

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🚀 AlphaTrader")
with col_h2:
    if not data.empty:
        last_date = data['run_date'].max().strftime('%d-%m')
        st.caption(f"Update: {last_date}")

if data.empty:
    st.warning("Wachten op data...")
    st.stop()

# ==========================================
# 2. LOGICA (SWEET SPOTS)
# ==========================================
def get_sweet_spots(df, lookahead_days, alpha_col, conf_col):
    if alpha_col not in df.columns or conf_col not in df.columns:
        return pd.DataFrame()

    # Filter signalen
    signals = df[(df[conf_col] > 70) & (df[alpha_col] > 1)].copy()
    results = []
    
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
            max_price = future_data['price'].max()
            days = (future_data['run_date'].max() - start_date).days
            status = "🏁 Klaar" if days >= lookahead_days else f"⏳ {days}d"
        else:
            max_price = start_price
            status = "🆕 Nieuw"
            
        pct = ((max_price - start_price) / start_price) * 100
        results.append({
            "Ticker": ticker,
            "Datum": start_date.strftime('%d-%m'),
            "Start": start_price,
            "Max": max_price,
            "Winst": pct,
            "Status": status
        })
    
    return pd.DataFrame(results)

# ==========================================
# 3. UI DEEL A: DE TABELLEN (BOVENAAN)
# ==========================================
st.markdown("### 🏆 Top Picks")

# Tabs werken perfect op mobiel
tab1, tab2 = st.tabs(["⚡ 2 Weken", "🔮 4 Weken"])

def show_table(tab, df_res):
    with tab:
        if not df_res.empty:
            avg = df_res['Winst'].mean()
            st.caption(f"Gem. Potentieel: **{avg:.2f}%**")
            
            # Mooiere tabel voor mobiel: minder kolommen, strakke formatting
            st.dataframe(
                df_res[['Ticker', 'Datum', 'Start', 'Winst', 'Status']].style.format({
                    "Start": "{:.2f}",
                    "Winst": "{:.2f}%"
                }).background_gradient(subset=['Winst'], cmap='Greens'),
                use_container_width=True,
                hide_index=True,
                height=300 # Scrollable container
            )
        else:
            st.info("Geen signalen.")

df_2w = get_sweet_spots(data, 21, 'alpha_2w_norm', 'confidence_2w')
df_4w = get_sweet_spots(data, 28, 'alpha_4w_norm', 'confidence_4w')

show_table(tab1, df_2w)
show_table(tab2, df_4w)

st.divider()

# ==========================================
# 4. UI DEEL B: CHART & ZOEKFUNCTIE
# ==========================================
st.markdown("### 🔎 Analyse")

# 1. Zoekfunctie (Alfabetisch gesorteerd)
tickers = sorted(data['ticker'].unique())
# Tip: Zet een index=None zodat hij leeg begint, of kies een default
default_idx = 0 if tickers else None
selected_ticker = st.selectbox("Zoek aandeel:", tickers, index=default_idx)

if selected_ticker:
    subset = data[data['ticker'] == selected_ticker]
    last_row = subset.iloc[-1]
    
    # 2. KPI Cards (Huidige status)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Prijs", f"{last_row['price']:.2f}")
    kpi2.metric("Alpha 2W", f"{last_row['alpha_2w_norm']:.2f}")
    kpi3.metric("Conf 2W", f"{last_row['confidence_2w']:.0f}%")

    # 3. De Chart (Subplot: Prijs boven, Signaal onder)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.5, 0.5], # 50% prijs, 50% signalen
        subplot_titles=(f"Prijsverloop {selected_ticker}", "AI Signalen")
    )

    # --- PANEL 1: PRIJS (Area Chart) ---
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['price'],
        name="Prijs",
        fill='tozeroy', # Vult de ruimte onder de lijn
        line=dict(color='#10B981', width=1), # Modern groen
        fillcolor='rgba(16, 185, 129, 0.1)' # Transparant groen
    ), row=1, col=1)

    # --- PANEL 2: SIGNALEN (Pastel lijnen) ---
    # Alpha (Links)
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['alpha_2w_norm'],
        name='Alpha 2W', line=dict(color='#5D9CEC', width=2) # Pastel Blauw
    ), row=2, col=1)
    
    # Conf (Rechts - via secundaire y-as truc in subplot)
    # Let op: Subplots met dubbele as zijn complex, we houden Conf en Alpha 
    # hier even op 1 as OF we normaliseren ze.
    # Beter voor mobiel leesbaarheid: Zet Conf als stippellijn erbij, 
    # maar we weten dat schaal anders is. 
    # TRUC: We zetten Conf op een 2e Y-as in de 2e rij.
    
    fig.add_trace(go.Scatter(
        x=subset['run_date'], y=subset['confidence_2w'],
        name='Conf 2W', line=dict(color='#ED5565', width=2, dash='dot') # Pastel Rood
    ), row=2, col=1, secondary_y=True)

    # Layout tuning voor mobiel
    fig.update_layout(
        height=500, # Niet te hoog op mobiel
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        showlegend=False, # Legende neemt ruimte in, hover is beter
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
    )
    
    # Assen netjes maken
    fig.update_yaxes(title_text="Prijs", row=1, col=1, showgrid=True, gridcolor='#F0F2F6')
    fig.update_yaxes(title_text="Alpha", row=2, col=1, showgrid=True, gridcolor='#F0F2F6')
    
    # We moeten de tweede y-as voor rij 2 handmatig toevoegen in layout
    fig.update_layout(
        yaxis3=dict(
            anchor="x", overlaying="y2", side="right", 
            title="Conf %", range=[0, 100], 
            showgrid=False, title_font=dict(color="#ED5565"), tickfont=dict(color="#ED5565")
        )
    )
    # Koppel de Conf trace aan deze yaxis3 (Plotly quirks)
    fig.data[2].update(yaxis="y3")

    st.plotly_chart(fig, use_container_width=True)
