from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dashboard Analisis Sentimen MyPertamina",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_ORDER = ["Negatif", "Positif"]
SENTIMENT_COLORS = {
    "Negatif": "#C64537",
    "Positif": "#238B5A",
}
DEFAULT_PREDICTION_PATH = Path("data/predictions/predictions_revisi_2kelas_20260421.csv")
DEFAULT_EVAL_SUMMARY_PATH = Path("logs/evaluation_summary_revisi_2kelas_20260421.json")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f5f8fc;
        --surface: #ffffff;
        --surface-soft: #f8fbfe;
        --border: #d8e2ec;
        --text: #14253d;
        --muted: #617489;
        --soft: #7f90a3;
        --sidebar: #eef4fa;
        --accent: #1c446f;
        --accent-soft: #edf4fb;
        --shadow-sm: 0 6px 18px rgba(18, 36, 62, 0.05);
        --shadow-md: 0 12px 30px rgba(18, 36, 62, 0.08);
        --radius-xl: 24px;
        --radius-lg: 18px;
        --radius-md: 14px;
        --negative: #C64537;
        --positive: #238B5A;
    }

    .stApp {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: radial-gradient(circle at top left, #ffffff 0%, #f7faff 40%, #edf3f9 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(249, 251, 254, 0.92);
        border-bottom: 1px solid var(--border);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar) 0%, #ebf1f8 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * { color: var(--text) !important; }

    .main .block-container {
        max-width: 1240px;
        padding-top: 1.05rem;
        padding-bottom: 1.35rem;
    }

    .hero-wrap {
        background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 1.15rem 1.45rem 1rem 1.45rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 0.9rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid #d5e4f3;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        font-size: 2.28rem;
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text);
        margin: 0 0 0.28rem 0;
    }

    .hero-subtitle {
        margin: 0;
        font-size: 0.99rem;
        color: var(--muted);
        font-weight: 500;
        line-height: 1.55;
    }

    .section-title {
        font-size: 1.72rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text);
        margin: 0.15rem 0 0.18rem 0;
    }

    .section-copy {
        font-size: 0.97rem;
        color: var(--muted);
        margin: 0 0 0.75rem 0;
        line-height: 1.55;
    }

    .mini-insight-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.25rem 0 0.95rem 0;
    }

    .mini-insight {
        background: linear-gradient(180deg, #fbfdff 0%, #f3f8fd 100%);
        border: 1px solid #d8e4ef;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        box-shadow: var(--shadow-sm);
    }

    .mini-insight-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        color: var(--accent);
        margin-bottom: 0.4rem;
    }

    .mini-insight-text {
        font-size: 0.96rem;
        line-height: 1.55;
        color: var(--text);
        margin: 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        margin-top: 0.15rem;
    }

    .stTabs [role="tab"] {
        border-radius: 999px !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid transparent !important;
        color: var(--muted) !important;
        font-weight: 600 !important;
        background: transparent !important;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        background: #eef5fc !important;
        border: 1px solid #d3e3f1 !important;
        color: var(--text) !important;
        box-shadow: inset 0 -2px 0 rgba(198,69,55,0.95);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.82rem 0.95rem;
        box-shadow: var(--shadow-sm);
    }

    .metric-primary [data-testid="stMetric"] {
        background: linear-gradient(180deg, #fcfdff 0%, #eef5fb 100%);
        border: 1px solid #d1dfed;
        box-shadow: 0 10px 22px rgba(28, 68, 111, 0.08);
    }

    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 700; margin-bottom: 0.2rem !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 800; letter-spacing: -0.03em; line-height: 1 !important; margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stMetricDelta"] { color: var(--soft) !important; }
    [data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] label, [data-testid="stMetricValue"] p {
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div {
        background: #ffffff !important;
        border: 1px solid #c7d3df !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }

    /* Fix warna dropdown filter Streamlit multiselect */
    div[data-baseweb="select"] * {
        color: #c64537 !important;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="select"] input::placeholder,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color: #c64537 !important;
        fill: #c64537 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="tag"] {
        background: #ffe9e5 !important;
        border: 1px solid #f3b8af !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] svg {
        color: #c64537 !important;
        fill: #c64537 !important;
    }

    div[data-baseweb="popover"] {
        z-index: 99999 !important;
    }

    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background: #ffffff !important;
    }

    div[data-baseweb="popover"] > div {
        border: 1px solid #e6c9c4 !important;
        border-radius: 14px !important;
        box-shadow: 0 14px 30px rgba(198, 69, 55, 0.12) !important;
    }

    ul[role="listbox"] {
        background: #ffffff !important;
        padding: 8px !important;
    }

    div[role="option"],
    ul[role="listbox"] li {
        background: #ffffff !important;
        color: #c64537 !important;
        border-radius: 10px !important;
    }

    div[role="option"] *,
    ul[role="listbox"] li *,
    div[role="option"] svg,
    ul[role="listbox"] svg {
        color: #c64537 !important;
        fill: #c64537 !important;
        opacity: 1 !important;
    }

    div[role="option"]:hover,
    ul[role="listbox"] li:hover {
        background: #fff3f1 !important;
        color: #c64537 !important;
    }

    div[role="option"][aria-selected="true"] {
        background: #ffe9e5 !important;
        color: #c64537 !important;
    }

    div[data-baseweb="popover"] div[aria-selected="true"] *,
    div[data-baseweb="popover"] div[role="option"] *,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] p {
        color: #c64537 !important;
    }

    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: var(--shadow-sm);
        padding: 0.2rem;
    }

    .note-box {
        background: linear-gradient(180deg, #f6fafe 0%, #eef4fb 100%);
        border: 1px solid #d7e3ef;
        border-radius: 16px;
        padding: 0.92rem 1rem;
        color: var(--text);
        font-size: 0.96rem;
        line-height: 1.62;
        margin-top: 0.6rem;
    }

    .legend-note {
        display: inline-flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 0.1rem 0 0.65rem 0;
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        background: #f8fbfe;
        border: 1px solid #dde7f1;
        width: fit-content;
    }

    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.92rem;
        color: var(--text);
        font-weight: 600;
    }

    .legend-dot {
        width: 11px;
        height: 11px;
        border-radius: 999px;
        display: inline-block;
    }

    .divider-soft {
        height: 1px;
        background: linear-gradient(90deg, rgba(216,226,236,0) 0%, rgba(216,226,236,1) 18%, rgba(216,226,236,1) 82%, rgba(216,226,236,0) 100%);
        margin: 0.95rem 0 1rem 0;
    }

    .rows-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    }

    .rows-label {
        font-size: 0.94rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.55rem;
    }

    .segmented-wrap {
        display: inline-flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-bottom: 0.95rem;
    }

    .segmented-wrap button {
        min-width: 74px;
        height: 44px;
        border-radius: 12px;
        border: 1px solid #c9d6e3;
        background: #f5f9fd;
        color: #1c3049;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 2px 6px rgba(20,37,61,0.05);
    }

    .segmented-wrap button:hover {
        border-color: #b3c6d9;
        background: #edf4fb;
        color: #10243b;
    }

    .segmented-wrap button.active {
        border: 1px solid #d94f42;
        background: linear-gradient(180deg, #ff6a5c 0%, #f05345 100%);
        color: #ffffff;
        box-shadow: 0 8px 18px rgba(198,69,55,0.22);
    }

    .table-shell {
        background: linear-gradient(180deg, #15181e 0%, #101319 100%);
        border: 1px solid #232a34;
        border-radius: 18px;
        box-shadow: 0 14px 30px rgba(11, 18, 30, 0.16);
        overflow: hidden;
    }

    .table-wrap {
        max-height: 520px;
        overflow: auto;
    }

    table.pred-table {
        width: 100%;
        border-collapse: collapse;
        color: #eef2f6;
        font-size: 0.95rem;
    }

    table.pred-table thead th {
        position: sticky;
        top: 0;
        z-index: 2;
        background: #1c2129;
        color: #cdd7e2;
        text-align: left;
        padding: 0.95rem 0.95rem;
        font-weight: 600;
        border-bottom: 1px solid #2a313c;
        white-space: nowrap;
    }

    table.pred-table tbody td {
        padding: 0.88rem 0.95rem;
        border-bottom: 1px solid #232a34;
        vertical-align: top;
    }

    table.pred-table tbody tr:nth-child(even) td {
        background: rgba(255,255,255,0.015);
    }

    table.pred-table tbody tr:hover td {
        background: rgba(255,255,255,0.04);
    }

    .td-content {
        min-width: 420px;
        max-width: 620px;
        line-height: 1.55;
        color: #f2f5f8;
    }

    .td-date, .td-score, .td-confidence {
        white-space: nowrap;
        color: #e6ebf1;
    }

    .sentiment-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.68rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    .pill-negative {
        background: #ffe3df;
        color: #7d2017;
        border: 1px solid #efb0a7;
        box-shadow: none;
    }

    .pill-positive {
        background: #ddf5e9;
        color: #145a39;
        border: 1px solid #9fd8bc;
        box-shadow: none;
    }

    [data-testid="stBaseButton-secondary"] {
        border-radius: 999px !important;
        border: 1px solid #c7d4e2 !important;
        background: #ffffff !important;
        color: #14253d !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        min-height: 44px !important;
        box-shadow: 0 2px 6px rgba(20,37,61,0.05) !important;
        opacity: 1 !important;
    }

    [data-testid="stBaseButton-secondary"] p,
    [data-testid="stBaseButton-secondary"] span,
    [data-testid="stBaseButton-secondary"] div {
        color: #14253d !important;
        opacity: 1 !important;
    }

    [data-testid="stBaseButton-secondary"]:hover {
        border-color: #d94f42 !important;
        background: linear-gradient(180deg, #ff6a5c 0%, #f05345 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 18px rgba(198,69,55,0.22) !important;
    }

    [data-testid="stBaseButton-secondary"]:hover p,
    [data-testid="stBaseButton-secondary"]:hover span,
    [data-testid="stBaseButton-secondary"]:hover div {
        color: #ffffff !important;
    }

    [data-testid="stBaseButton-primary"] {
        border-radius: 999px !important;
        border: 1px solid #d94f42 !important;
        background: linear-gradient(180deg, #ff6a5c 0%, #f05345 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 18px rgba(198,69,55,0.22) !important;
        opacity: 1 !important;
    }

    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span,
    [data-testid="stBaseButton-primary"] div {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, li,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption {
        color: var(--text) !important;
    }

    @media (max-width: 900px) {
        .mini-insight-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.8rem; }
        .td-content { min-width: 280px; max-width: 420px; }
    }
</style>
""",
    unsafe_allow_html=True,
)


def section_header(title: str, copy: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if copy:
        st.markdown(f"<p class='section-copy'>{copy}</p>", unsafe_allow_html=True)


def chart_legend_note() -> None:
    st.markdown(
        """
        <div class='legend-note'>
            <span class='legend-item'><span class='legend-dot' style='background:#C64537;'></span>Merah = Sentimen Negatif</span>
            <span class='legend-item'><span class='legend-dot' style='background:#238B5A;'></span>Hijau = Sentimen Positif</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def polish(fig: go.Figure, height: int = 360, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={"l": 20, "r": 20, "t": 56, "b": 24},
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#14253d"},
        legend_title_text="Sentimen",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#14253d"},
            "bgcolor": "rgba(255,255,255,0.92)",
        },
        showlegend=show_legend,
        title={"font": {"color": "#14253d", "size": 18}},
        hoverlabel={"font": {"color": "#14253d"}},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e6edf5", zeroline=False, tickfont={"color": "#14253d"}, title_font={"color": "#14253d"})
    fig.update_yaxes(showgrid=True, gridcolor="#e6edf5", zeroline=False, tickfont={"color": "#14253d"}, title_font={"color": "#14253d"})
    return fig


@st.cache_data(show_spinner=False)
def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "label_name" not in df.columns:
        raise ValueError("Kolom label_name tidak ditemukan pada file prediksi")
    df["label_name"] = df["label_name"].astype("string").str.strip().str.title()
    if "at" in df.columns:
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
        df["year"] = df["at"].dt.year
        df["month"] = df["at"].dt.to_period("M").astype("string")
    else:
        df["year"] = pd.NA
        df["month"] = pd.NA
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_eval_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filter Data")
        st.caption("Atur subset data untuk membaca pola sentimen dan detail prediksi.")
        labels = [s for s in SENTIMENT_ORDER if s in df["label_name"].dropna().unique()]
        selected_labels = st.multiselect("Sentimen", options=labels, default=labels)
        years = sorted(df["year"].dropna().astype(int).unique().tolist()) if df["year"].notna().any() else []
        year_range = None
        if years:
            year_range = st.slider("Rentang Tahun", min_value=int(years[0]), max_value=int(years[-1]), value=(int(years[0]), int(years[-1])), step=1)
        st.caption(f"Data awal: {len(df):,} baris")

    filtered = df.copy()
    if selected_labels:
        filtered = filtered[filtered["label_name"].isin(selected_labels)]
    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
    return filtered


def build_insights(df: pd.DataFrame, summary: dict) -> tuple[str, str]:
    total = len(df)
    pos = int((df["label_name"] == "Positif").sum())
    pos_pct = (pos / total * 100) if total else 0
    yearly = df.dropna(subset=["year"]).groupby(["year", "label_name"], as_index=False).size().rename(columns={"size": "count"})
    trend_note = "Komposisi tahunan belum dapat dihitung."
    if not yearly.empty:
        pivot = yearly.pivot(index="year", columns="label_name", values="count").fillna(0)
        if "Positif" in pivot.columns:
            pct = (pivot["Positif"] / pivot.sum(axis=1) * 100).round(1)
            if len(pct) >= 2:
                trend_note = f"Proporsi sentimen positif berubah dari {pct.iloc[0]:.1f}% pada {int(pct.index[0])} menjadi {pct.iloc[-1]:.1f}% pada {int(pct.index[-1])}."
            else:
                trend_note = f"Proporsi sentimen positif pada periode yang tampil adalah {pct.iloc[0]:.1f}%."
    weighted_f1 = summary.get("metrics", {}).get("weighted_f1", 0)
    rec_pos = summary.get("classificationReport", {}).get("Positif", {}).get("recall", 0)
    return (
        f"Dari {total:,} ulasan yang tampil, sentimen negatif masih mendominasi dengan porsi positif {pos_pct:.1f}% saja.",
        f"{trend_note} Model final tetap kuat dengan weighted F1 {weighted_f1:.4f}, tetapi recall positif masih {rec_pos:.4f}."
    )


def render_overview(df: pd.DataFrame, summary: dict) -> None:
    section_header("Ringkasan Sentimen", "Halaman ini dirancang agar pembaca langsung menangkap komposisi sentimen, arah tren, dan kualitas model klasifikasi secara cepat.")
    note_a, note_b = build_insights(df, summary)
    st.markdown(f"""
        <div class='mini-insight-grid'>
            <div class='mini-insight'><div class='mini-insight-label'>Insight utama</div><p class='mini-insight-text'>{note_a}</p></div>
            <div class='mini-insight'><div class='mini-insight-label'>Makna penelitian</div><p class='mini-insight-text'>{note_b}</p></div>
        </div>
    """, unsafe_allow_html=True)

    total = len(df)
    negatif = int((df["label_name"] == "Negatif").sum())
    positif = int((df["label_name"] == "Positif").sum())
    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns and df["confidence"].notna().any() else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ulasan", f"{total:,}")
    c2.metric("Negatif", f"{negatif:,}")
    c3.metric("Positif", f"{positif:,}")
    c4.metric("Avg Confidence", f"{avg_conf:.3f}")

    chart_legend_note()

    dist = df["label_name"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).rename_axis("label_name").reset_index(name="count")
    left, right = st.columns([1, 1.55])
    with left:
        pie = px.pie(dist, names="label_name", values="count", color="label_name", color_discrete_map=SENTIMENT_COLORS, hole=0.6, title="Distribusi Sentimen")
        pie.update_traces(textposition="outside", textinfo="percent+label", marker={"line": {"color": "#ffffff", "width": 3}}, pull=[0.02, 0.01], sort=False, showlegend=True)
        pie.update_layout(legend={"orientation": "h", "yanchor": "top", "y": 1.12, "xanchor": "left", "x": 0, "font": {"color": "#14253d"}})
        st.plotly_chart(polish(pie, 355, show_legend=True), use_container_width=True)
    with right:
        if df["month"].notna().any():
            monthly = df.dropna(subset=["month"]).groupby(["month", "label_name"], as_index=False).size().rename(columns={"size": "count"})
            trend = px.line(monthly, x="month", y="count", color="label_name", markers=True, category_orders={"label_name": SENTIMENT_ORDER}, color_discrete_map=SENTIMENT_COLORS, title="Tren Sentimen per Bulan")
            trend.update_traces(line={"width": 2.8}, marker={"size": 6}, showlegend=True)
            trend.update_layout(xaxis_title="Periode", yaxis_title="Jumlah Ulasan")
            st.plotly_chart(polish(trend, 355, show_legend=True), use_container_width=True)
            st.caption("Catatan: lonjakan tajam di awal periode dipengaruhi volume ulasan yang sangat tinggi pada tahun 2022.")
        else:
            st.info("Data bulan tidak tersedia")


def render_yearly_distribution(df: pd.DataFrame) -> None:
    section_header("Penyebaran Positif vs Negatif per Tahun", "Perbandingan jumlah dan proporsi tahunan membantu membaca perubahan persepsi pengguna secara lebih adil antar periode.")
    if not df["year"].notna().any():
        st.info("Kolom tahun tidak tersedia")
        return
    chart_legend_note()
    yearly = df.dropna(subset=["year"]).groupby(["year", "label_name"], as_index=False).size().rename(columns={"size": "count"})
    yearly["year"] = yearly["year"].astype(int)
    yearly = yearly[yearly["label_name"].isin(SENTIMENT_ORDER)]
    left, right = st.columns(2)
    with left:
        fig_count = px.bar(yearly, x="year", y="count", color="label_name", barmode="group", category_orders={"label_name": SENTIMENT_ORDER}, color_discrete_map=SENTIMENT_COLORS, text_auto=True, title="Jumlah Ulasan per Tahun")
        fig_count.update_traces(textposition="outside", cliponaxis=False, showlegend=True)
        fig_count.update_layout(xaxis_title="Tahun", yaxis_title="Jumlah Ulasan")
        st.plotly_chart(polish(fig_count, 410, show_legend=True), use_container_width=True)
    with right:
        yearly_pct = yearly.copy()
        yearly_pct["pct"] = yearly_pct.groupby("year")["count"].transform(lambda s: s / s.sum() * 100)
        fig_pct = px.bar(yearly_pct, x="year", y="pct", color="label_name", barmode="stack", category_orders={"label_name": SENTIMENT_ORDER}, color_discrete_map=SENTIMENT_COLORS, text_auto=".1f", title="Proporsi Sentimen per Tahun (%)")
        fig_pct.update_traces(texttemplate="%{y:.1f}%", showlegend=True)
        fig_pct.update_layout(xaxis_title="Tahun", yaxis_title="Persentase")
        st.plotly_chart(polish(fig_pct, 410, show_legend=True), use_container_width=True)


def render_evaluation(summary: dict) -> None:
    section_header("Evaluasi Model 2 Kelas", "Bagian ini menjelaskan seberapa baik model IndoBERT membedakan ulasan positif dan negatif pada konfigurasi final penelitian.")
    if not summary:
        st.warning("Ringkasan evaluasi tidak ditemukan")
        return
    metrics = summary.get("metrics", {})
    cls_report = summary.get("classificationReport", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro F1", f"{metrics.get('macro_f1', 0):.4f}")
    c2.metric("Weighted F1", f"{metrics.get('weighted_f1', 0):.4f}")
    c3.metric("Recall Negatif", f"{cls_report.get('Negatif', {}).get('recall', 0):.4f}")
    c4.metric("Recall Positif", f"{cls_report.get('Positif', {}).get('recall', 0):.4f}")
    cm_payload = summary.get("confusionMatrix", {})
    if cm_payload:
        cm_df = pd.DataFrame(cm_payload).T.fillna(0)
        for col in cm_df.columns:
            cm_df[col] = pd.to_numeric(cm_df[col], errors="coerce").fillna(0).astype(int)
        left, right = st.columns([1.28, 0.82])
        with left:
            fig_cm = go.Figure(data=go.Heatmap(z=cm_df.values, x=list(cm_df.columns), y=list(cm_df.index), colorscale=[[0, "#edf3fa"], [0.42, "#bfd3ea"], [1, "#1c446f"]], text=cm_df.values, texttemplate="%{text}", textfont={"size": 14, "color": "#14253d"}, showscale=True))
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Prediksi", yaxis_title="Aktual")
            st.plotly_chart(polish(fig_cm, 398, show_legend=False), use_container_width=True)
        with right:
            st.markdown("""
                <div class='note-box'>
                    <strong>Interpretasi cepat.</strong><br><br>
                    Model final sudah kuat secara umum, terlihat dari weighted F1 yang tinggi. Namun performa antar kelas belum sepenuhnya seimbang: kelas negatif lebih mudah dikenali dibanding kelas positif. Artinya, pembacaan insight positif tetap perlu disertai kehati-hatian karena sebagian ulasan positif masih dapat bergeser ke prediksi negatif.
                </div>
            """, unsafe_allow_html=True)
    st.markdown("""
        <div class='note-box'>
            <strong>Catatan metodologis.</strong> Konfigurasi final memakai 2 kelas karena distribusi data tidak seimbang. Skema 3 kelas diposisikan sebagai eksperimen pembanding, bukan model final.
        </div>
    """, unsafe_allow_html=True)


def render_custom_table(df: pd.DataFrame, rows: int) -> None:
    view = df.copy()
    if "at" in view.columns:
        view = view.sort_values("at", ascending=False)
    cols = [c for c in ["at", "content", "score", "label_name", "confidence"] if c in view.columns]
    view = view[cols].head(rows).copy()

    if "at" in view.columns:
        view["at"] = view["at"].dt.strftime("%Y-%m-%d %H:%M")
    if "score" in view.columns:
        view["score"] = view["score"].apply(lambda x: "" if pd.isna(x) else str(int(x)) if float(x).is_integer() else f"{x}")
    if "confidence" in view.columns:
        view["confidence"] = view["confidence"].apply(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")

    headers = {
        "at": "at",
        "content": "content",
        "score": "score",
        "label_name": "label_name",
        "confidence": "confidence",
    }

    thead = "".join(f"<th>{html.escape(headers[col])}</th>" for col in cols)
    rows_html = []
    for _, row in view.iterrows():
        cells = []
        for col in cols:
            value = "" if pd.isna(row[col]) else str(row[col])
            safe_value = html.escape(value)
            if col == "label_name":
                cls = "pill-positive" if value.lower() == "positif" else "pill-negative"
                cells.append(f"<td><span class='sentiment-pill {cls}'>{safe_value}</span></td>")
            elif col == "content":
                cells.append(f"<td class='td-content'>{safe_value}</td>")
            elif col == "at":
                cells.append(f"<td class='td-date'>{safe_value}</td>")
            elif col == "score":
                cells.append(f"<td class='td-score'>{safe_value}</td>")
            elif col == "confidence":
                cells.append(f"<td class='td-confidence'>{safe_value}</td>")
            else:
                cells.append(f"<td>{safe_value}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class='table-shell'>
        <div class='table-wrap'>
            <table class='pred-table'>
                <thead><tr>{thead}</tr></thead>
                <tbody>{''.join(rows_html)}</tbody>
            </table>
        </div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_table(df: pd.DataFrame) -> None:
    section_header("Data Prediksi", "Tabel ini dipakai untuk memeriksa contoh ulasan, label hasil prediksi, dan confidence model pada level data individual.")
    st.markdown("<div class='rows-label'>Jumlah baris</div>", unsafe_allow_html=True)

    if "table_rows" not in st.session_state:
        st.session_state.table_rows = 25

    options = [10, 25, 50, 100]
    button_cols = st.columns([0.07, 0.07, 0.07, 0.07, 0.72])
    for idx, value in enumerate(options):
        with button_cols[idx]:
            active = st.session_state.table_rows == value
            if st.button(str(value), key=f"rows_{value}", use_container_width=True, type="secondary"):
                st.session_state.table_rows = value

    render_custom_table(df, st.session_state.table_rows)


def main() -> None:
    st.markdown("""
        <div class='hero-wrap'>
            <div class='eyebrow'>Dashboard Penelitian</div>
            <h1 class='hero-title'>Dashboard Analisis Sentiment MyPertamina</h1>
            <p class='hero-subtitle'>Oleh Rofiq Roihan — visualisasi hasil klasifikasi sentimen, tren ulasan, dan evaluasi model final IndoBERT.</p>
        </div>
    """, unsafe_allow_html=True)

    base_dir = Path(__file__).resolve().parents[1]
    prediction_path = base_dir / DEFAULT_PREDICTION_PATH
    summary_path = base_dir / DEFAULT_EVAL_SUMMARY_PATH

    if not prediction_path.exists():
        st.error(f"File prediksi tidak ditemukan: {prediction_path}")
        st.stop()

    try:
        df_all = load_predictions(prediction_path)
    except Exception as exc:
        st.error(f"Gagal memuat data prediksi: {exc}")
        st.stop()

    summary = load_eval_summary(summary_path)
    df_filtered = render_filters(df_all)
    if df_filtered.empty:
        st.warning("Tidak ada data sesuai filter yang dipilih")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Overview", "Chart Tahunan", "Data"])
    with tab1:
        render_overview(df_filtered, summary)
        st.markdown("<div class='divider-soft'></div>", unsafe_allow_html=True)
        render_evaluation(summary)
    with tab2:
        render_yearly_distribution(df_filtered)
    with tab3:
        render_table(df_filtered)


if __name__ == "__main__":
    main()