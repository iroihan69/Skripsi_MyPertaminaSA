from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard Sentimen MyPertamina",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

pio.templates.default = "plotly_white"

# ============================================================
# KONFIGURASI & STYLING
# ============================================================

SENTIMENT_ORDER = ["Negatif", "Netral", "Positif"]
SENTIMENT_COLORS = {
    "Negatif": "#CB5A43",
    "Netral": "#7F8C8D",
    "Positif": "#1F8B5F",
}

SENTIMENT_EMOJI = {
    "Negatif": "😞",
    "Netral": "😐",
    "Positif": "😊",
}

# Custom CSS untuk styling dashboard modern dan ringkas
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #ecf2ee;
        --text-main: #1f2a24;
        --accent: #1f8b5f;
        --line: rgba(31, 42, 36, 0.14);
        --surface-soft: #f5faf7;
        --muted: #5a6a60;
    }

    .stApp {
        font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;
        background: radial-gradient(circle at 8% 5%, #dfece3 0%, #ecf2ee 44%, #f2f6f3 100%);
        color: var(--text-main);
    }

    .main .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        background: #f9fcfa;
        border: 1px solid #d5e2d9;
        border-radius: 22px;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
        box-shadow: 0 10px 30px rgba(22, 41, 30, 0.07);
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: #edf3ef;
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5fbf7 0%, #edf4ef 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--text-main);
    }

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stCaption,
    label,
    p,
    span,
    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        color: var(--text-main);
    }

    [data-testid="stTabs"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 8px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(22, 41, 30, 0.05);
    }

    [data-testid="stTabs"] button {
        color: var(--text-main);
        border-radius: 10px !important;
        border: 1px solid transparent !important;
        border-bottom: 2px solid transparent !important;
        transition: background-color 0.18s ease, border-color 0.18s ease;
        box-shadow: none !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text-main);
        background: #e8f4ec;
        border: 1px solid #b8d8c3 !important;
        border-bottom: 2px solid #1f8b5f !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }

    [data-testid="stTabs"] button::after,
    [data-testid="stTabs"] button::before {
        display: none !important;
    }

    .hero-card {
        background: linear-gradient(125deg, #1f8b5f 0%, #2aa36f 100%);
        border: 1px solid #1d7f57;
        padding: 18px 20px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 12px;
        box-shadow: 0 10px 20px rgba(25, 84, 56, 0.2);
    }

    .hero-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        font-size: 0.92rem;
        opacity: 0.95;
        color: #e6f6ee;
        font-weight: 500;
    }

    .quick-stat {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
        color: var(--text-main);
        box-shadow: 0 3px 10px rgba(22, 41, 30, 0.06);
    }

    .quick-stat strong {
        color: var(--text-main);
    }

    .quick-stat span,
    .quick-stat div,
    .quick-stat p {
        color: var(--text-main);
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d3e2d8;
        border-radius: 12px;
        padding: 8px 12px;
        box-shadow: 0 3px 10px rgba(22, 41, 30, 0.06);
        transition: box-shadow 0.18s ease;
    }

    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 14px rgba(21, 61, 41, 0.12);
    }

    .stButton > button {
        background: #e8f4ec;
        color: var(--text-main);
        border: 1px solid #c0d8c9;
        border-radius: 10px;
        transition: background-color 0.18s ease, border-color 0.18s ease;
    }

    .stButton > button:hover {
        background: #d8ebdf;
        border-color: #98c1a7;
        color: var(--text-main);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff !important;
        color: #1f2a24 !important;
        border: 1px solid #c3d7c9 !important;
    }

    div[data-baseweb="select"] svg,
    div[data-baseweb="input"] svg,
    div[data-baseweb="base-input"] svg {
        fill: #1f2a24 !important;
        color: #1f2a24 !important;
    }

    div[data-baseweb="popover"] * {
        color: #1f2a24 !important;
        background: #ffffff !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: var(--text-main);
    }

    [data-testid="stMetricDelta"] {
        background: #e7f6ec;
        border-radius: 999px;
        padding: 2px 8px;
        width: fit-content;
        border: 1px solid #bfdfcb;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #cfe0d5;
        border-radius: 10px;
        background: #ffffff;
        box-shadow: 0 3px 10px rgba(22, 41, 30, 0.05);
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid #d0e0d5;
        border-radius: 12px;
        padding: 6px;
        background: #ffffff;
        box-shadow: 0 3px 10px rgba(22, 41, 30, 0.05);
    }

    [data-testid="stPlotlyChart"] text {
        fill: #1f2a24 !important;
    }

    [data-testid="stDataFrame"] div,
    [data-testid="stTable"] div {
        color: var(--text-main);
    }

    [data-testid="stAlertContainer"] {
        color: var(--text-main);
    }

    .stCaption,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {
        color: var(--muted) !important;
        opacity: 1 !important;
    }

    .stSelectbox > div,
    .stMultiSelect > div,
    .stTextInput > div,
    .stDateInput > div,
    .stNumberInput > div,
    .stSlider,
    .stCheckbox,
    .stRadio {
        color: var(--text-main);
    }
</style>
""", unsafe_allow_html=True)


def normalize_cm_label(label: str) -> str:
    """Normalize confusion-matrix label agar cocok antara axis aktual dan prediksi."""
    text = str(label).strip()
    text = re.sub(r"^(aktual|prediksi)\s+", "", text, flags=re.IGNORECASE)
    return text.title()


def polish_figure(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply konsistensi visual chart agar tampilan dashboard lebih profesional."""
    title_text = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Plus Jakarta Sans, Segoe UI, sans-serif", "color": "#1f2a24", "size": 13},
        margin={"l": 18, "r": 18, "t": 42, "b": 18},
        legend=dict(bgcolor="rgba(255,255,255,0.7)"),
        title={"text": title_text, "font": {"color": "#1f2a24", "size": 24}},
        legend_font={"color": "#1f2a24", "size": 13},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e8efe9",
        zeroline=False,
        title_font={"color": "#2c3b32", "size": 16},
        tickfont={"color": "#2c3b32", "size": 13},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e8efe9",
        zeroline=False,
        title_font={"color": "#2c3b32", "size": 16},
        tickfont={"color": "#2c3b32", "size": 13},
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig

# ============================================================
# FUNGSI UTILITY
# ============================================================

@st.cache_data(show_spinner=False)
def load_predictions(csv_path: str) -> pd.DataFrame:
    """Load dan prepare dataset dengan kolom tambahan untuk filtering."""
    df = pd.read_csv(csv_path)

    if "at" in df.columns:
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
        df["year"] = df["at"].dt.year
        df["month"] = df["at"].dt.to_period("M").astype("string")
        df["date"] = df["at"].dt.date
    else:
        df["year"] = pd.NA
        df["month"] = pd.NA
        df["date"] = pd.NA

    if "label_name" in df.columns:
        df["label_name"] = (
            df["label_name"]
            .astype("string")
            .str.strip()
            .str.title()
            .fillna("Tidak Diketahui")
        )

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_confusion_matrix(summary_path: str) -> pd.DataFrame:
    """Load confusion matrix dari evaluation summary."""
    summary_file = Path(summary_path)
    if not summary_file.exists():
        return pd.DataFrame()

    with summary_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    cm_data = payload.get("confusionMatrix")
    if not cm_data:
        return pd.DataFrame()

    cm_df = pd.DataFrame(cm_data).T.fillna(0)
    for col in cm_df.columns:
        cm_df[col] = pd.to_numeric(cm_df[col], errors="coerce").fillna(0).astype(int)
    return cm_df


def extract_words(text: str) -> list[str]:
    """Extract words dari teks untuk wordcloud."""
    if pd.isna(text):
        return []
    # Remove non-alphabetic characters, lowercase, split
    words = re.findall(r"\b[a-z]+\b", str(text).lower())
    # Simple stopwords untuk bahasa Indonesia
    stopwords_id = {
        "dan", "yang", "di", "ke", "untuk", "dari", "dengan", "adalah",
        "tidak", "ada", "atau", "ini", "itu", "pada", "telah", "dapat",
        "juga", "akan", "seperti", "oleh", "sudah", "kalau", "karena",
        "menjadi", "saat", "tapi", "namun", "sesuai", "tanpa", "kali",
        "sering", "hanya", "lebih", "dalam", "nya", "an", "a", "i",
        "u", "e", "o", "ya", "ga", "gak", "iya", "lah", "nih", "deh"
    }
    return [w for w in words if w not in stopwords_id and len(w) > 2]


def create_wordcloud(texts: list[str], sentiment: str) -> plt.Figure:
    """Generate wordcloud untuk sebuah sentimen."""
    text_combined = " ".join(texts)
    
    if not text_combined.strip():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Data tidak tersedia", ha="center", va="center")
        ax.axis("off")
        return fig

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="RdYlGn" if sentiment == "Positif" else ("Reds" if sentiment == "Negatif" else "Blues"),
        max_words=100,
    ).generate(text_combined)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Hitung metrik agregat untuk dataset."""
    total = len(df)
    neg_count = int((df["label_name"] == "Negatif").sum())
    neu_count = int((df["label_name"] == "Netral").sum())
    pos_count = int((df["label_name"] == "Positif").sum())

    neg_pct = (neg_count / total * 100) if total > 0 else 0
    neu_pct = (neu_count / total * 100) if total > 0 else 0
    pos_pct = (pos_count / total * 100) if total > 0 else 0

    avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0

    return {
        "total": total,
        "negatif_count": neg_count,
        "netral_count": neu_count,
        "positif_count": pos_count,
        "negatif_pct": neg_pct,
        "netral_pct": neu_pct,
        "positif_pct": pos_pct,
        "avg_confidence": avg_confidence,
    }


def format_compact_count(value: int) -> str:
    """Format angka menjadi ringkas agar tidak terpotong pada KPI card."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,}"


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar dengan filter interaktif yang lebih baik."""
    with st.sidebar:
        st.header("🎛️ Filter Data")
        
        # Tab-based filter organization
        tab1, tab2, tab3 = st.tabs(["Sentimen", "Waktu", "Advanced"])
        
        with tab1:
            st.subheader("Sentimen")
            sentiment_options = [s for s in SENTIMENT_ORDER if s in df["label_name"].dropna().unique()]
            
            # Better sentiment selection dengan emoji
            all_sentiments = st.checkbox("Semua Sentimen", value=True, key="all_sent")
            
            if all_sentiments:
                selected_labels = sentiment_options
            else:
                selected_labels = []
                for sentiment in sentiment_options:
                    if st.checkbox(f"{SENTIMENT_EMOJI.get(sentiment, '•')} {sentiment}", value=True, key=f"sent_{sentiment}"):
                        selected_labels.append(sentiment)
            
            if not selected_labels:
                st.warning("Pilih minimal satu sentimen!")
                selected_labels = sentiment_options
        
        with tab2:
            st.subheader("Rentang Waktu")
            
            has_date = "at" in df.columns and df["at"].notna().any()
            
            if has_date:
                available_years = sorted(df[df["year"].notna()]["year"].astype(int).unique().tolist())
                
                if available_years:
                    # Year range slider - lebih intuitif
                    year_range = st.slider(
                        "Pilih Rentang Tahun",
                        min_value=int(available_years[0]),
                        max_value=int(available_years[-1]),
                        value=(int(available_years[0]), int(available_years[-1])),
                        step=1,
                        key="year_range"
                    )
                    
                    # Month range di dalam tahun
                    st.subheader("Filter Bulan")
                    show_month_filter = st.checkbox(
                        "Filter bulan tertentu?",
                        value=False,
                        key="show_month_filter",
                    )
                    
                    if show_month_filter:
                        available_months = sorted(
                            df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]["month"]
                            .dropna()
                            .unique()
                            .tolist()
                        )
                        selected_months = st.multiselect(
                            "Pilih bulan(s)",
                            options=available_months,
                            default=available_months,
                            key="selected_months",
                        )
                    else:
                        selected_months = None
                else:
                    year_range = None
                    selected_months = None
            else:
                st.info("Kolom tanggal tidak tersedia")
                year_range = None
                selected_months = None
        
        with tab3:
            st.subheader("Advanced Filters")
            
            # Confidence filter
            use_confidence = st.checkbox(
                "Filter berdasarkan confidence?",
                value=False,
                key="use_confidence",
            )
            confidence_range = None
            if use_confidence and "confidence" in df.columns:
                confidence_range = st.slider(
                    "Minimum Confidence Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key="confidence_min",
                )
            
            # Rating filter
            use_rating = st.checkbox(
                "Filter berdasarkan rating?",
                value=False,
                key="use_rating",
            )
            rating_range = None
            if use_rating and "score" in df.columns:
                rating_range = st.slider(
                    "Rentang Rating",
                    min_value=1,
                    max_value=5,
                    value=(1, 5),
                    step=1,
                    key="rating_range",
                )
        
        # Filter summary
        st.divider()
        
        # Apply filters
        filtered_df = df.copy()
        
        # Sentiment filter
        if selected_labels:
            filtered_df = filtered_df[filtered_df["label_name"].isin(selected_labels)]
        
        # Year range filter
        if year_range:
            filtered_df = filtered_df[
                (
                    (filtered_df["year"] >= year_range[0])
                    & (filtered_df["year"] <= year_range[1])
                )
                | filtered_df["year"].isna()
            ]
        
        # Month filter
        if selected_months:
            filtered_df = filtered_df[filtered_df["month"].isin(selected_months)]
        
        # Confidence filter
        if confidence_range is not None:
            filtered_df = filtered_df[filtered_df["confidence"] >= confidence_range]
        
        # Rating filter
        if rating_range is not None:
            filtered_df = filtered_df[
                (filtered_df["score"] >= rating_range[0]) & 
                (filtered_df["score"] <= rating_range[1])
            ]
        
        # Show filtered count
        st.markdown(f"""
        **📊 Status Filter:**
        - Data yang ditampilkan: {len(filtered_df):,} dari {len(df):,} ulasan
        - Persentase: {len(filtered_df)/len(df)*100:.1f}%
        """)
        
        # Clear filters button
        if st.button("🔄 Reset Semua Filter", use_container_width=True):
            filter_keys = [
                "all_sent",
                "show_month_filter",
                "selected_months",
                "year_range",
                "use_confidence",
                "confidence_min",
                "use_rating",
                "rating_range",
            ]
            filter_keys.extend([f"sent_{sentiment}" for sentiment in SENTIMENT_ORDER])

            for key in filter_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    return filtered_df


def render_summary_insights(df_filtered: pd.DataFrame, df_all: pd.DataFrame) -> None:
    """Render ringkasan eksekutif yang sederhana dan mudah dipindai."""
    st.subheader("Ringkasan Utama", divider="rainbow")
    
    metrics = calculate_metrics(df_filtered)
    metrics_all = calculate_metrics(df_all)
    
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Intelligence Snapshot</div>
            <div class="hero-subtitle">Ringkasan cepat sentimen pengguna berdasarkan filter aktif.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        positif_delta = metrics['positif_pct'] - metrics_all['positif_pct']
        st.metric(
            f"{SENTIMENT_EMOJI['Positif']} Positif",
            f"{format_compact_count(metrics['positif_count'])}",
            delta=f"{metrics['positif_pct']:.1f}% | {positif_delta:+.1f} pp"
        )
    
    with col2:
        netral_delta = metrics['netral_pct'] - metrics_all['netral_pct']
        st.metric(
            f"{SENTIMENT_EMOJI['Netral']} Netral",
            f"{format_compact_count(metrics['netral_count'])}",
            delta=f"{metrics['netral_pct']:.1f}% | {netral_delta:+.1f} pp"
        )
    
    with col3:
        negatif_delta = metrics['negatif_pct'] - metrics_all['negatif_pct']
        st.metric(
            f"{SENTIMENT_EMOJI['Negatif']} Negatif",
            f"{format_compact_count(metrics['negatif_count'])}",
            delta=f"{metrics['negatif_pct']:.1f}% | {negatif_delta:+.1f} pp"
        )
    
    with col4:
        st.metric(
            "🎯 Avg. Confidence",
            f"{metrics['avg_confidence']:.3f}",
            delta="Kualitas prediksi"
        )

    dominant = "Positif" if metrics['positif_pct'] >= metrics['negatif_pct'] else "Negatif"
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="quick-stat"><strong>Sentimen Dominan:</strong> {dominant}</div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="quick-stat"><strong>Data Aktif:</strong> {metrics['total']:,} ulasan</div>
            """,
            unsafe_allow_html=True,
        )


def render_sentiment_overview(df: pd.DataFrame) -> None:
    """Render visual utama yang ringkas dan interaktif."""
    st.subheader("Peta Sentimen", divider="blue")
    
    distribution = (
        df["label_name"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("label_name")
        .reset_index(name="count")
    )
    
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        st.markdown("Komposisi Sentimen")
        pie_fig = px.pie(
            distribution,
            names="label_name",
            values="count",
            color="label_name",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.55,
        )
        pie_fig.update_traces(textposition="inside", textinfo="percent+label")
        pie_fig.update_layout(showlegend=False)
        polish_figure(pie_fig, height=340)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        st.markdown("Volume per Sentimen")
        bar_fig = px.bar(
            distribution,
            x="count",
            y="label_name",
            color="label_name",
            color_discrete_map=SENTIMENT_COLORS,
            text_auto=True,
        )
        bar_fig.update_layout(
            showlegend=False,
            xaxis_title="Jumlah Ulasan",
            yaxis_title="",
        )
        bar_fig.update_traces(textposition="outside")
        polish_figure(bar_fig, height=340)
        st.plotly_chart(bar_fig, use_container_width=True)

    has_month = "month" in df.columns and df["month"].notna().any()
    if has_month:
        trend_df = (
            df.dropna(subset=["month"])
            .groupby(["month", "label_name"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        if not trend_df.empty:
            trend_fig = px.area(
                trend_df,
                x="month",
                y="count",
                color="label_name",
                color_discrete_map=SENTIMENT_COLORS,
                category_orders={"label_name": SENTIMENT_ORDER},
            )
            trend_fig.update_layout(
                title="Pergerakan Sentimen per Bulan",
                xaxis_title="Periode",
                yaxis_title="Jumlah Ulasan",
                legend_title="",
            )
            polish_figure(trend_fig, height=360)
            st.plotly_chart(trend_fig, use_container_width=True)


def render_detailed_analysis(df: pd.DataFrame) -> None:
    """Render analisis detail untuk pengguna lanjutan."""
    st.subheader("Analisis Lanjutan", divider="green")
    
    # Tabs for different detailed views
    tab1, tab2, tab3, tab4 = st.tabs(["Tren Temporal", "Confidence Analysis", "Rating vs Sentimen", "Text Analytics"])
    
    with tab1:
        st.markdown("**Tren Sentimen per Bulan**")
        
        has_month = "month" in df.columns and df["month"].notna().any()
        if not has_month:
            st.info("Kolom tanggal tidak tersedia untuk analisis tren")
        else:
            trend_df = (
                df.dropna(subset=["month"])
                .groupby(["month", "label_name"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )

            if not trend_df.empty:
                trend_fig = px.line(
                    trend_df,
                    x="month",
                    y="count",
                    color="label_name",
                    color_discrete_map=SENTIMENT_COLORS,
                    markers=True,
                    category_orders={"label_name": SENTIMENT_ORDER},
                    title="Perkembangan Sentimen Per Bulan"
                )
                trend_fig.update_layout(
                    xaxis_title="Periode",
                    yaxis_title="Jumlah Ulasan",
                    legend_title="Sentimen",
                )
                polish_figure(trend_fig, height=400)
                st.plotly_chart(trend_fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Confidence Score Distribution**")
        
        if "confidence" in df.columns and df["confidence"].notna().any():
            # Distribution histogram
            conf_fig = px.histogram(
                df,
                x="confidence",
                nbins=50,
                title="Sebaran Confidence Score Model",
                labels={"confidence": "Confidence Score", "count": "Jumlah Prediksi"}
            )
            conf_fig.update_layout(height=400)
            polish_figure(conf_fig, height=400)
            st.plotly_chart(conf_fig, use_container_width=True)
            
            # Stats per sentimen
            conf_stats = df.groupby("label_name")["confidence"].agg([
                ("Mean", "mean"),
                ("Median", "median"),
                ("Std Dev", "std"),
                ("Min", "min"),
                ("Max", "max"),
            ]).round(4)
            
            st.markdown("**Statistik Confidence per Sentimen**")
            st.dataframe(conf_stats, use_container_width=True)
        else:
            st.warning("Kolom confidence tidak tersedia")
    
    with tab3:
        st.markdown("**Hubungan Rating (Score) vs Sentimen Prediksi**")
        
        if "score" in df.columns:
            rating_sent = (
                df.groupby(["score", "label_name"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            
            rating_fig = px.bar(
                rating_sent,
                x="score",
                y="count",
                color="label_name",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="group",
                title="Distribusi Rating per Sentimen Prediksi",
                labels={"score": "Rating", "count": "Jumlah Ulasan"}
            )
            rating_fig.update_layout(height=400)
            polish_figure(rating_fig, height=400)
            st.plotly_chart(rating_fig, use_container_width=True)
            
            # Cross-tabulation
            st.markdown("**Cross-Tabulation: Rating vs Sentimen**")
            crosstab = pd.crosstab(df["score"], df["label_name"], margins=True)
            st.dataframe(crosstab, use_container_width=True)
        else:
            st.warning("Kolom rating tidak tersedia")
    
    with tab4:
        st.markdown("**Statistik Teks**")
        
        df_temp = df.copy()
        df_temp["text_length"] = df_temp["content"].str.len()
        df_temp["word_count"] = df_temp["content"].str.split().str.len()
        
        text_stats = df_temp.groupby("label_name").agg({
            "text_length": ["mean", "min", "max"],
            "word_count": ["mean", "min", "max"],
        }).round(2)
        
        st.markdown("**Metrik Teks per Sentimen**")
        st.dataframe(text_stats, use_container_width=True)
        
        # Length distribution
        col1, col2 = st.columns(2)
        
        with col1:
            len_fig = px.box(
                df_temp,
                y="label_name",
                x="text_length",
                color="label_name",
                color_discrete_map=SENTIMENT_COLORS,
                title="Distribusi Panjang Teks",
                labels={"text_length": "Jumlah Karakter"}
            )
            len_fig.update_layout(showlegend=False, height=400)
            polish_figure(len_fig, height=400)
            st.plotly_chart(len_fig, use_container_width=True)
        
        with col2:
            word_fig = px.box(
                df_temp,
                y="label_name",
                x="word_count",
                color="label_name",
                color_discrete_map=SENTIMENT_COLORS,
                title="Distribusi Jumlah Kata",
                labels={"word_count": "Jumlah Kata"}
            )
            word_fig.update_layout(showlegend=False, height=400)
            polish_figure(word_fig, height=400)
            st.plotly_chart(word_fig, use_container_width=True)


def render_wordcloud_analysis(df: pd.DataFrame) -> None:
    """Render wordcloud interaktif agar user fokus ke satu sudut pandang sekaligus."""
    st.subheader("Eksplorasi Kata", divider="violet")

    selected_sentiment = st.radio(
        "Pilih sentimen:",
        options=SENTIMENT_ORDER,
        horizontal=True,
    )

    texts = df[df["label_name"] == selected_sentiment]["content"].dropna().tolist()
    if not texts:
        st.info("Belum ada data untuk sentimen ini.")
        return

    words = []
    for text in texts:
        words.extend(extract_words(text))

    if not words:
        st.info("Tidak ada kata yang cukup untuk divisualisasikan.")
        return

    top_words = Counter(words).most_common(12)
    top_words_df = pd.DataFrame(top_words, columns=["kata", "frekuensi"])

    col_wc, col_terms = st.columns([2, 1])
    with col_wc:
        try:
            fig_wc = create_wordcloud(words, selected_sentiment)
            st.pyplot(fig_wc, use_container_width=True)
        except Exception as e:
            st.warning(f"Tidak bisa generate wordcloud: {str(e)}")

    with col_terms:
        terms_fig = px.bar(
            top_words_df.sort_values("frekuensi"),
            x="frekuensi",
            y="kata",
            orientation="h",
            title="Top Keywords",
            color_discrete_sequence=["#5b7cfa"],
        )
        terms_fig.update_layout(yaxis_title="", xaxis_title="", showlegend=False)
        polish_figure(terms_fig, height=380)
        st.plotly_chart(terms_fig, use_container_width=True)


def render_model_evaluation(cm_df: pd.DataFrame) -> None:
    """Render evaluasi model dengan confusion matrix dan metrik."""
    st.subheader("🎯 Evaluasi Model", divider="orange")
    
    if cm_df.empty:
        st.info("Confusion matrix tidak tersedia pada file evaluasi")
        return
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("**Confusion Matrix**")
        display_cm = cm_df.copy()
        display_cm.index = [normalize_cm_label(lbl) for lbl in display_cm.index]
        display_cm.columns = [normalize_cm_label(lbl) for lbl in display_cm.columns]

        heatmap = go.Figure(
            data=go.Heatmap(
                z=display_cm.values,
                x=display_cm.columns,
                y=display_cm.index,
                colorscale="Blues",
                text=display_cm.values,
                texttemplate="%{text}",
                textfont={"size": 14},
            )
        )
        heatmap.update_layout(
            xaxis_title="Prediksi Model",
            yaxis_title="Label Aktual",
        )
        polish_figure(heatmap, height=400)
        st.plotly_chart(heatmap, use_container_width=True)
    
    with col2:
        st.markdown("Ringkasan Model")

        row_map = {normalize_cm_label(label): label for label in cm_df.index}
        col_map = {normalize_cm_label(label): label for label in cm_df.columns}
        class_labels = [label for label in SENTIMENT_ORDER if label in row_map and label in col_map]

        if not class_labels:
            st.warning("Format confusion matrix tidak dikenali untuk perhitungan metrik.")
            return
        
        # Calculate metrics dari confusion matrix
        total = cm_df.values.sum()
        correct = sum(cm_df.loc[row_map[label], col_map[label]] for label in class_labels)
        accuracy = correct / total if total > 0 else 0
        
        st.metric("Accuracy", f"{accuracy:.2%}", "test set")
        st.markdown("---")
        
        # Per-class metrics
        st.markdown("F1 per Kelas")
        for label in class_labels:
            row_label = row_map[label]
            col_label = col_map[label]

            tp = cm_df.loc[row_label, col_label]
            fp = cm_df[col_label].sum() - tp
            fn = cm_df.loc[row_label].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            st.caption(f"{label}: {f1:.3f}")


def render_prediction_table(df: pd.DataFrame) -> None:
    """Render tabel data prediksi dengan detail."""
    st.subheader("📋 Tabel Data Prediksi", divider="gray")
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        sort_by = st.selectbox(
            "Urutkan berdasarkan:",
            ["Terbaru", "Tertua", "Confidence Tertinggi", "Confidence Terendah"]
        )
    
    with col2:
        rows_displayed = st.selectbox("Baris yang ditampilkan:", [10, 25, 50, 100])
    
    with col3:
        st.write("")  # spacing
    
    # Apply sorting
    display_df = df.copy()
    
    if sort_by == "Terbaru" and "at" in display_df.columns:
        display_df = display_df.sort_values("at", ascending=False)
    elif sort_by == "Tertua" and "at" in display_df.columns:
        display_df = display_df.sort_values("at", ascending=True)
    elif sort_by == "Confidence Tertinggi" and "confidence" in display_df.columns:
        display_df = display_df.sort_values("confidence", ascending=False)
    elif sort_by == "Confidence Terendah" and "confidence" in display_df.columns:
        display_df = display_df.sort_values("confidence", ascending=True)
    
    display_df = display_df.head(rows_displayed)
    
    # Select columns
    display_columns = [
        col
        for col in ["at", "content", "score", "label_name", "confidence"]
        if col in display_df.columns
    ]
    
    preview_df = display_df[display_columns].copy()
    
    if "at" in preview_df.columns:
        preview_df["at"] = preview_df["at"].dt.strftime("%Y-%m-%d %H:%M")
    if "confidence" in preview_df.columns:
        preview_df["confidence"] = preview_df["confidence"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    # Rename columns untuk display lebih rapi
    preview_df = preview_df.rename(columns={
        "at": "Tanggal",
        "content": "Ulasan",
        "score": "Rating",
        "label_name": "Sentimen",
        "confidence": "Confidence"
    })

    st.dataframe(preview_df, use_container_width=True, hide_index=True, height=450)
    st.caption(f"Menampilkan {len(preview_df)} dari {len(df):,} ulasan dengan filter aktif")


def main() -> None:
    """Main application entry point."""
    st.title("Sentiment Intelligence | MyPertamina")
    st.caption("Dashboard interaktif untuk membaca pola sentimen pengguna secara cepat")
    
    # Load data
    base_dir = Path(__file__).resolve().parents[1]
    prediction_path = base_dir / "data" / "predictions" / "predictions.csv"
    eval_summary_path = base_dir / "logs" / "evaluation_summary.json"
    
    if not prediction_path.exists():
        st.error(f"❌ File prediksi tidak ditemukan: {prediction_path}")
        st.stop()
    
    df_all = load_predictions(str(prediction_path))
    
    required_columns = {"content", "label_name", "confidence"}
    missing_columns = sorted(required_columns.difference(df_all.columns))
    if missing_columns:
        st.error(f"❌ Kolom wajib tidak lengkap pada predictions.csv: {', '.join(missing_columns)}")
        st.stop()
    
    # Apply filters from sidebar
    df_filtered = render_sidebar_filters(df_all)
    
    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data sesuai filter saat ini. Coba ubah filter Anda.")
        st.stop()
    
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "Overview",
        "Analisis Lanjutan",
        "Kata dan Evaluasi",
        "Data Explorer",
    ])
    
    with main_tab1:
        render_summary_insights(df_filtered, df_all)
        st.divider()
        render_sentiment_overview(df_filtered)
    
    with main_tab2:
        render_detailed_analysis(df_filtered)
    
    with main_tab3:
        render_wordcloud_analysis(df_filtered)
        st.divider()
        render_model_evaluation(load_confusion_matrix(str(eval_summary_path)))

    with main_tab4:
        render_prediction_table(df_filtered)
    
    # Footer dengan info teknis
    st.divider()
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.caption(f"📊 Total Data: {len(df_all):,} ulasan")
    
    with col_info2:
        if "at" in df_all.columns and df_all["at"].notna().any():
            date_range = f"{df_all['at'].min().date()} hingga {df_all['at'].max().date()}"
            st.caption(f"📅 Periode: {date_range}")
    
    with col_info3:
        st.caption("🤖 Model: IndoBERT (Fine-tuned untuk Sentimen 3-kelas)")


if __name__ == "__main__":
    main()
