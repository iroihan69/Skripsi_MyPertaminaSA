from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sentiment Analytics — MyPertamina",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SENTIMENT_ORDER = ["Negatif", "Positif"]
SENTIMENT_COLORS = {
    "Negatif": "#E0524B",
    "Positif": "#1F9D6B",
}
DEFAULT_PREDICTION_PATH = Path("data/predictions/predictions_exp07_2kelas.csv")
DEFAULT_EVAL_SUMMARY_PATH = Path("logs/evaluation_summary_exp07_2kelas.json")

# ════════════════════════════════════════════════════════════════════════════
# STYLING — "Analytics Console": dark rail sidebar + premium light canvas
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
        --bg: #f4f6f9;
        --panel: #ffffff;
        --ink: #0f172a;
        --ink-soft: #5a6678;
        --ink-faint: #8b94a4;
        --line: #e8ebf0;
        --line-soft: #eef1f5;
        --accent: #4f46e5;
        --accent-2: #6366f1;
        --accent-soft: #eef2ff;
        --neg: #E0524B;
        --pos: #1F9D6B;
        --neg-soft: #fdeeed;
        --pos-soft: #e8f7f0;
        --side-1: #1b2335;
        --side-2: #141b29;
        --side-ink: #e7ebf3;
        --side-soft: #9aa6bd;
        --shadow-sm: 0 4px 14px rgba(15, 23, 42, 0.05);
        --shadow-md: 0 14px 36px rgba(15, 23, 42, 0.09);
        --shadow-lg: 0 24px 60px rgba(15, 23, 42, 0.12);
        --r-lg: 22px;
        --r-md: 16px;
        --r-sm: 12px;
    }

    .stApp {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background:
            radial-gradient(1200px 600px at 100% -5%, #eef1f7 0%, rgba(238,241,247,0) 55%),
            var(--bg);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
        border-bottom: none;
        box-shadow: none;
    }
    /* keep the header action icons (menu / sidebar toggle) visible on light canvas */
    [data-testid="stHeader"] button[kind="header"] svg,
    [data-testid="stHeader"] [data-testid="stToolbar"] svg { color: var(--ink-soft); fill: var(--ink-soft); }

    .main .block-container {
        max-width: 1280px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* default text in main canvas */
    h1, h2, h3, h4, h5, h6, p, span, label, li,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption {
        color: var(--ink);
    }

    /* ───────────────────────── SIDEBAR (dark rail) ───────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(195deg, var(--side-1) 0%, var(--side-2) 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .stMarkdown { color: var(--side-ink) !important; }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--side-soft) !important;
    }

    .side-brand {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0 0.1rem 1rem;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .side-logo {
        width: 38px; height: 38px; border-radius: 11px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        display: grid; place-items: center;
        font-weight: 900; color: #fff; font-size: 0.95rem;
        box-shadow: 0 8px 18px rgba(79, 70, 229, 0.4);
    }
    .side-brand-text { line-height: 1.2; }
    .side-brand-title { font-weight: 800; font-size: 0.96rem; color: #fff; letter-spacing: -0.01em; }
    .side-brand-sub { font-size: 0.72rem; color: var(--side-soft); }

    .side-section {
        font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--side-soft);
        margin: 1.1rem 0 0.4rem;
    }

    /* pills inside sidebar — target exact Streamlit button testids */
    [data-testid="stSidebar"] [data-testid="stBaseButton-pills"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: #cdd5e4 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-pills"] p { color: #cdd5e4 !important; }
    [data-testid="stSidebar"] [data-testid="stBaseButton-pills"]:hover {
        background: rgba(255,255,255,0.11) !important;
        border-color: rgba(255,255,255,0.25) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-pills"]:hover p { color: #ffffff !important; }
    [data-testid="stSidebar"] [data-testid="stBaseButton-pillsActive"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
        border: 1px solid transparent !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px rgba(79,70,229,0.35) !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-pillsActive"] p { color: #ffffff !important; }

    /* expander on dark */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { color: var(--side-ink) !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] svg { fill: var(--side-soft) !important; }

    /* sidebar status card */
    .side-status {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        margin: 1rem 0 0.7rem;
    }
    .side-status-big { font-size: 1.45rem; font-weight: 800; color: #fff; line-height: 1; }
    .side-status-sub { font-size: 0.8rem; color: var(--side-soft); margin-top: 0.25rem; }
    .side-status-bar { height: 6px; border-radius: 999px; background: rgba(255,255,255,0.1); margin-top: 0.6rem; overflow: hidden; }
    .side-status-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%); }

    /* sidebar reset button */
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #e7ebf3 !important;
        border-radius: 11px !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.25) !important;
        color: #fff !important;
    }

    /* sidebar slider accent */
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent-2) !important;
    }

    /* ───────────────────────── TOP BAR (hero) ────────────────────────────── */
    .topbar {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafd 100%);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        padding: 1.35rem 1.6rem;
        margin-top: 0.2rem;
        margin-bottom: 1.1rem;
        display: flex; align-items: center; justify-content: space-between;
        gap: 1rem; flex-wrap: wrap;
        box-shadow: var(--shadow-md);
        position: relative; overflow: hidden;
    }
    .topbar::after {
        content: ''; position: absolute; right: -70px; top: -90px;
        width: 280px; height: 280px; border-radius: 50%;
        background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, rgba(99,102,241,0) 70%);
    }
    .topbar-left { display: flex; align-items: center; gap: 0.95rem; z-index: 1; }
    .topbar-logo {
        width: 50px; height: 50px; border-radius: 14px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
        display: grid; place-items: center; font-weight: 900; color: #fff; font-size: 1.15rem;
        box-shadow: 0 10px 24px rgba(79,70,229,0.32);
    }
    .topbar-title { font-size: 1.5rem; font-weight: 800; color: var(--ink) !important; letter-spacing: -0.02em; line-height: 1.15; }
    .topbar-sub { font-size: 0.9rem; color: var(--ink-soft) !important; font-weight: 500; margin-top: 0.15rem; }
    .topbar-right { display: flex; gap: 0.5rem; flex-wrap: wrap; z-index: 1; }
    .chip {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: var(--bg);
        border: 1px solid var(--line);
        color: var(--ink-soft) !important;
        padding: 0.42rem 0.85rem; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600;
    }
    .chip-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px rgba(34,197,94,0.7); }

    /* ───────────────────────── SECTION HEADERS ───────────────────────────── */
    .section-title { font-size: 1.28rem; font-weight: 800; letter-spacing: -0.02em; color: var(--ink); margin: 0.2rem 0 0.15rem; }
    .section-copy { font-size: 0.92rem; color: var(--ink-soft); margin: 0 0 0.9rem; line-height: 1.55; }

    /* ───────────────────────── HEALTH STRIP ──────────────────────────────── */
    .health {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        padding: 1.2rem 1.4rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.9rem;
    }
    .health-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .health-eyebrow { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-faint); }
    .health-dominant { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; color: var(--ink); margin-top: 0.25rem; }
    .health-meta { font-size: 0.88rem; color: var(--ink-soft); margin-top: 0.15rem; }
    .health-legend { display: flex; gap: 1.1rem; align-items: center; }
    .leg { display: inline-flex; align-items: center; gap: 0.45rem; font-size: 0.85rem; font-weight: 600; color: var(--ink-soft); }
    .leg-dot { width: 10px; height: 10px; border-radius: 3px; }
    .health-bar { display: flex; height: 46px; border-radius: 12px; overflow: hidden; box-shadow: inset 0 0 0 1px rgba(15,23,42,0.04); }
    .health-seg { display: grid; place-items: center; color: #fff; font-weight: 800; font-size: 0.95rem; transition: width 0.4s ease; min-width: 42px; }

    /* ───────────────────────── KPI CARDS ─────────────────────────────────── */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.85rem; margin-bottom: 1.1rem; }
    .kcard {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        padding: 1.05rem 1.15rem 1.1rem;
        box-shadow: var(--shadow-sm);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .kcard:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
    .kcard-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.7rem; }
    .kcard-label { font-size: 0.74rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-faint); }
    .kcard-ic { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; font-size: 0.9rem; }
    .ic-total { background: var(--accent-soft); }
    .ic-neg { background: var(--neg-soft); }
    .ic-pos { background: var(--pos-soft); }
    .ic-conf { background: #eef6ff; }
    .kcard-value { font-size: 1.95rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; color: var(--ink); }
    .kcard.neg .kcard-value { color: var(--neg); }
    .kcard.pos .kcard-value { color: var(--pos); }
    .kcard-sub { font-size: 0.82rem; color: var(--ink-soft); margin: 0.45rem 0 0; }
    .kbar { height: 5px; border-radius: 999px; background: var(--line-soft); margin-top: 0.7rem; overflow: hidden; }
    .kbar-fill { height: 100%; border-radius: 999px; }

    /* ───────────────────────── CHART PANELS ──────────────────────────────── */
    [data-testid="stPlotlyChart"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        box-shadow: var(--shadow-sm);
        padding: 0.45rem 0.55rem 0.2rem;
    }
    .panel-label { font-size: 0.95rem; font-weight: 700; color: var(--ink); margin: 0.4rem 0 0.1rem; }

    /* ───────────────────────── NOTE BOX ──────────────────────────────────── */
    .note-box {
        background: linear-gradient(180deg, #fbfcff 0%, #f4f7fd 100%);
        border: 1px solid var(--line);
        border-left: 3px solid var(--accent);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        color: var(--ink); font-size: 0.93rem; line-height: 1.62;
        margin-top: 0.6rem;
    }
    .note-box strong { color: var(--accent); }

    /* ───────────────────────── TABS ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem; background: var(--panel);
        border: 1px solid var(--line); border-radius: 14px;
        padding: 0.35rem; box-shadow: var(--shadow-sm);
        margin-bottom: 1.1rem;
    }
    .stTabs [role="tab"] {
        border-radius: 10px !important; padding: 0.55rem 1.2rem !important;
        border: none !important; color: var(--ink-soft) !important;
        font-weight: 600 !important; font-size: 0.9rem !important;
        background: transparent !important; transition: all 0.15s ease !important;
    }
    .stTabs [role="tab"]:hover:not([aria-selected="true"]) { background: var(--bg) !important; color: var(--ink) !important; }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
        color: #fff !important; box-shadow: 0 6px 16px rgba(79,70,229,0.3) !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { background: transparent !important; display: none !important; }

    /* ───────────────────────── METRICS (eval tab) ────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: var(--r-md); padding: 0.9rem 1rem;
        box-shadow: var(--shadow-sm);
    }
    [data-testid="stMetricLabel"] { color: var(--ink-faint) !important; font-weight: 700 !important; text-transform: uppercase; font-size: 0.72rem !important; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; letter-spacing: -0.03em; }

    /* segmented control on canvas — target exact Streamlit button testids */
    [data-testid="stBaseButton-segmented_control"] {
        background: #ffffff !important;
        color: var(--ink-soft) !important;
        border: 1px solid var(--line) !important;
        font-weight: 600 !important;
    }
    [data-testid="stBaseButton-segmented_control"] p { color: var(--ink-soft) !important; }
    [data-testid="stBaseButton-segmented_control"]:hover {
        background: var(--accent-soft) !important;
        border-color: #c7ccf7 !important;
        color: var(--accent) !important;
    }
    [data-testid="stBaseButton-segmented_control"]:hover p { color: var(--accent) !important; }

    [data-testid="stBaseButton-segmented_controlActive"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
        color: #ffffff !important;
        border: 1px solid transparent !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px rgba(79,70,229,0.28) !important;
    }
    [data-testid="stBaseButton-segmented_controlActive"] p { color: #ffffff !important; }
    [data-testid="stBaseButton-segmented_controlActive"]:hover {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%) !important;
        color: #ffffff !important;
    }
    [data-testid="stBaseButton-segmented_controlActive"]:hover p { color: #ffffff !important; }

    /* ───────────────────────── DATA TABLE (dark terminal) ────────────────── */
    .table-shell {
        background: linear-gradient(180deg, #161a23 0%, #0f131b 100%);
        border: 1px solid #232a36; border-radius: 18px;
        box-shadow: var(--shadow-md); overflow: hidden; margin-top: 0.8rem;
    }
    .table-wrap { max-height: 540px; overflow: auto; }
    table.pred-table { width: 100%; border-collapse: collapse; color: #eef2f6; font-size: 0.93rem; }
    table.pred-table thead th {
        position: sticky; top: 0; z-index: 2;
        background: #1b212c; color: #aeb9c9; text-align: left;
        padding: 0.9rem 0.95rem; font-weight: 700; font-size: 0.74rem;
        text-transform: uppercase; letter-spacing: 0.06em;
        border-bottom: 1px solid #2a313c; white-space: nowrap;
    }
    table.pred-table tbody td { padding: 0.82rem 0.95rem; border-bottom: 1px solid #20262f; vertical-align: top; }
    table.pred-table tbody tr:nth-child(even) td { background: rgba(255,255,255,0.014); }
    table.pred-table tbody tr:hover td { background: rgba(99,102,241,0.07); }
    .td-content { min-width: 420px; max-width: 640px; line-height: 1.55; color: #eef2f7; }
    .td-date, .td-score, .td-confidence { white-space: nowrap; color: #c8d0dc; }
    .sentiment-pill { display: inline-flex; align-items: center; padding: 0.28rem 0.66rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
    .pill-negative { background: rgba(224,82,75,0.16); color: #ff9b94; border: 1px solid rgba(224,82,75,0.4); }
    .pill-positive { background: rgba(31,157,107,0.16); color: #5fd9a6; border: 1px solid rgba(31,157,107,0.4); }
    .conf-mini { display: inline-flex; align-items: center; gap: 0.5rem; color: #e6ebf2; }
    .conf-track { width: 46px; height: 6px; border-radius: 999px; background: rgba(255,255,255,0.12); overflow: hidden; flex: 0 0 auto; }
    .conf-bar { height: 100%; background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%); border-radius: 999px; }
    .conf-val { color: #eef2f8 !important; font-weight: 700; font-variant-numeric: tabular-nums; font-size: 0.92rem; }

    @media (max-width: 900px) {
        .kpi-row { grid-template-columns: repeat(2, 1fr); }
        .topbar-title { font-size: 1.2rem; }
        .td-content { min-width: 260px; max-width: 380px; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def section_header(title: str, copy: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if copy:
        st.markdown(f"<p class='section-copy'>{copy}</p>", unsafe_allow_html=True)


def panel_label(text: str) -> None:
    st.markdown(f"<div class='panel-label'>{text}</div>", unsafe_allow_html=True)


def polish(fig: go.Figure, height: int = 360, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin={"l": 16, "r": 16, "t": 50, "b": 20},
        font={"family": "Inter, Segoe UI, sans-serif", "size": 13, "color": "#0f172a"},
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1,
            "font": {"color": "#0f172a", "size": 12},
            "bgcolor": "rgba(255,255,255,0)",
        },
        showlegend=show_legend,
        legend_title_text=None,
        title={"text": ""},
        hoverlabel={"font": {"color": "#0f172a", "family": "Inter"}, "bgcolor": "#ffffff", "bordercolor": "#e8ebf0"},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont={"color": "#5a6678"}, title_font={"color": "#5a6678", "size": 12})
    fig.update_yaxes(showgrid=True, gridcolor="#eef1f5", zeroline=False, tickfont={"color": "#5a6678"}, title_font={"color": "#5a6678", "size": 12})
    return fig


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ════════════════════════════════════════════════════════════════════════════
def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown(
            """
            <div class='side-brand'>
                <div class='side-logo'>SA</div>
                <div class='side-brand-text'>
                    <div class='side-brand-title'>Sentiment Analytics</div>
                    <div class='side-brand-sub'>MyPertamina · IndoBERT</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='side-section'>Filter Data</div>", unsafe_allow_html=True)

        labels = [s for s in SENTIMENT_ORDER if s in df["label_name"].dropna().unique()]
        selected_labels = st.pills(
            "Sentimen", options=labels, selection_mode="multi", default=labels, key="f_sent"
        )
        if not selected_labels:
            selected_labels = labels

        years = sorted(df["year"].dropna().astype(int).unique().tolist()) if df["year"].notna().any() else []
        selected_years = years
        if years:
            selected_years = st.pills(
                "Tahun", options=years, selection_mode="multi",
                default=years, format_func=lambda y: str(y), key="f_year",
            )
            if not selected_years:
                selected_years = years

        use_confidence = False
        conf_min = 0.0
        with st.expander("Filter Lanjutan"):
            if "confidence" in df.columns:
                use_confidence = st.checkbox("Aktifkan ambang confidence", value=False, key="use_conf")
                if use_confidence:
                    conf_min = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="conf_min")
            else:
                st.caption("Kolom confidence tidak tersedia.")

        # apply filters
        filtered = df.copy()
        if selected_labels:
            filtered = filtered[filtered["label_name"].isin(selected_labels)]
        if years and selected_years:
            filtered = filtered[filtered["year"].isin(selected_years)]
        if use_confidence and conf_min > 0.0:
            filtered = filtered[filtered["confidence"] >= conf_min]

        pct_shown = len(filtered) / len(df) * 100 if len(df) > 0 else 0
        st.markdown(
            f"""
            <div class='side-status'>
                <div class='side-status-big'>{len(filtered):,}</div>
                <div class='side-status-sub'>dari {len(df):,} total ulasan · {pct_shown:.1f}%</div>
                <div class='side-status-bar'><div class='side-status-fill' style='width:{pct_shown:.1f}%;'></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Reset Filter", use_container_width=True):
            for key in ["f_sent", "f_year", "use_conf", "conf_min", "table_sort", "table_rows"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    return filtered


# ════════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
def build_insights(df: pd.DataFrame, summary: dict) -> tuple[str, str]:
    total = len(df)
    pos = int((df["label_name"] == "Positif").sum())
    neg = int((df["label_name"] == "Negatif").sum())
    pos_pct = (pos / total * 100) if total else 0
    neg_pct = (neg / total * 100) if total else 0

    dominan = "positif" if pos_pct >= neg_pct else "negatif"
    dominan_pct = pos_pct if dominan == "positif" else neg_pct

    yearly = (
        df.dropna(subset=["year"])
        .groupby(["year", "label_name"], as_index=False)
        .size().rename(columns={"size": "count"})
    )
    trend_note = "Komposisi tahunan belum dapat dihitung."
    if not yearly.empty:
        pivot = yearly.pivot(index="year", columns="label_name", values="count").fillna(0)
        if "Positif" in pivot.columns:
            pct = (pivot["Positif"] / pivot.sum(axis=1) * 100).round(1)
            if len(pct) >= 2:
                trend_note = (
                    f"Proporsi sentimen positif bergeser dari {pct.iloc[0]:.1f}% pada "
                    f"{int(pct.index[0])} menjadi {pct.iloc[-1]:.1f}% pada {int(pct.index[-1])}."
                )
            else:
                trend_note = f"Proporsi sentimen positif pada periode yang tampil adalah {pct.iloc[0]:.1f}%."

    weighted_f1 = summary.get("metrics", {}).get("weighted_f1", 0)
    rec_pos = summary.get("classificationReport", {}).get("Positif", {}).get("recall", 0)

    return (
        f"Dari {total:,} ulasan yang tampil, sentimen {dominan} mendominasi dengan porsi {dominan_pct:.1f}%.",
        f"{trend_note} Model final memperoleh weighted F1 {weighted_f1:.4f} dengan recall positif {rec_pos:.4f}.",
    )


# ════════════════════════════════════════════════════════════════════════════
# HEALTH STRIP + KPI CARDS
# ════════════════════════════════════════════════════════════════════════════
def render_health_and_kpi(df: pd.DataFrame) -> None:
    total = len(df)
    neg = int((df["label_name"] == "Negatif").sum())
    pos = int((df["label_name"] == "Positif").sum())
    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns and df["confidence"].notna().any() else 0.0
    neg_pct = neg / total * 100 if total else 0.0
    pos_pct = pos / total * 100 if total else 0.0

    dominant = "Positif" if pos_pct >= neg_pct else "Negatif"
    dom_color = SENTIMENT_COLORS[dominant]
    dom_pct = pos_pct if dominant == "Positif" else neg_pct

    # health strip with 100% proportion bar
    neg_seg = f"<div class='health-seg' style='width:{neg_pct:.1f}%;background:{SENTIMENT_COLORS['Negatif']};'>{neg_pct:.0f}%</div>" if neg_pct >= 4 else f"<div class='health-seg' style='width:{neg_pct:.1f}%;background:{SENTIMENT_COLORS['Negatif']};'></div>"
    pos_seg = f"<div class='health-seg' style='width:{pos_pct:.1f}%;background:{SENTIMENT_COLORS['Positif']};'>{pos_pct:.0f}%</div>" if pos_pct >= 4 else f"<div class='health-seg' style='width:{pos_pct:.1f}%;background:{SENTIMENT_COLORS['Positif']};'></div>"

    st.markdown(
        f"""
        <div class='health'>
            <div class='health-top'>
                <div>
                    <div class='health-eyebrow'>Komposisi Sentimen</div>
                    <div class='health-dominant'>Sentimen <span style='color:{dom_color};'>{dominant}</span> mendominasi · {dom_pct:.1f}%</div>
                    <div class='health-meta'>Berdasarkan {total:,} ulasan pada filter aktif</div>
                </div>
                <div class='health-legend'>
                    <span class='leg'><span class='leg-dot' style='background:{SENTIMENT_COLORS['Negatif']};'></span>Negatif</span>
                    <span class='leg'><span class='leg-dot' style='background:{SENTIMENT_COLORS['Positif']};'></span>Positif</span>
                </div>
            </div>
            <div class='health-bar'>{neg_seg}{pos_seg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI cards
    st.markdown(
        f"""
        <div class='kpi-row'>
            <div class='kcard'>
                <div class='kcard-head'>
                    <span class='kcard-label'>Total Ulasan</span>
                    <span class='kcard-ic ic-total'>📄</span>
                </div>
                <div class='kcard-value'>{total:,}</div>
                <p class='kcard-sub'>Data ditampilkan pada filter</p>
            </div>
            <div class='kcard neg'>
                <div class='kcard-head'>
                    <span class='kcard-label'>Negatif</span>
                    <span class='kcard-ic ic-neg'>▼</span>
                </div>
                <div class='kcard-value'>{neg:,}</div>
                <p class='kcard-sub'>{neg_pct:.1f}% dari total</p>
                <div class='kbar'><div class='kbar-fill' style='width:{neg_pct:.1f}%;background:{SENTIMENT_COLORS['Negatif']};'></div></div>
            </div>
            <div class='kcard pos'>
                <div class='kcard-head'>
                    <span class='kcard-label'>Positif</span>
                    <span class='kcard-ic ic-pos'>▲</span>
                </div>
                <div class='kcard-value'>{pos:,}</div>
                <p class='kcard-sub'>{pos_pct:.1f}% dari total</p>
                <div class='kbar'><div class='kbar-fill' style='width:{pos_pct:.1f}%;background:{SENTIMENT_COLORS['Positif']};'></div></div>
            </div>
            <div class='kcard'>
                <div class='kcard-head'>
                    <span class='kcard-label'>Avg Confidence</span>
                    <span class='kcard-ic ic-conf'>◎</span>
                </div>
                <div class='kcard-value'>{avg_conf:.3f}</div>
                <p class='kcard-sub'>Skor kepercayaan model</p>
                <div class='kbar'><div class='kbar-fill' style='width:{avg_conf*100:.1f}%;background:var(--accent);'></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
def render_overview(df: pd.DataFrame, summary: dict) -> None:
    note_a, note_b = build_insights(df, summary)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class='note-box'><strong>Insight utama.</strong><br>{note_a}</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class='note-box'><strong>Makna penelitian.</strong><br>{note_b}</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    render_health_and_kpi(df)

    dist = (
        df["label_name"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("label_name").reset_index(name="count")
    )
    dist["label_name"] = dist["label_name"].astype(str)
    dist["count"] = dist["count"].astype(int)
    left, right = st.columns([1, 1.55])

    with left:
        panel_label("Distribusi Sentimen")
        pie = px.pie(
            dist, names="label_name", values="count", color="label_name",
            color_discrete_map=SENTIMENT_COLORS, hole=0.66,
        )
        pie.update_traces(
            textposition="outside", textinfo="percent+label",
            marker={"line": {"color": "#ffffff", "width": 3}},
            sort=False, showlegend=False,
        )
        st.plotly_chart(polish(pie, 340, show_legend=False), use_container_width=True)

    with right:
        panel_label("Tren Sentimen per Bulan")
        if df["month"].notna().any():
            monthly = (
                df.dropna(subset=["month"]).groupby(["month", "label_name"], as_index=False)
                .size().rename(columns={"size": "count"})
            )
            monthly["month"] = monthly["month"].astype(str)
            monthly["label_name"] = monthly["label_name"].astype(str)
            monthly["count"] = monthly["count"].astype(int)
            trend = px.area(
                monthly, x="month", y="count", color="label_name",
                category_orders={"label_name": SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_COLORS,
            )
            trend.update_traces(line={"width": 0}, opacity=0.85)
            trend.update_layout(xaxis_title="Periode", yaxis_title="Jumlah Ulasan")
            st.plotly_chart(polish(trend, 340, show_legend=True), use_container_width=True)
        else:
            st.info("Data bulan tidak tersedia")
    st.caption("Lonjakan volume di awal periode dipengaruhi tingginya aktivitas ulasan pada tahun 2022.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUSI & KORELASI
# ════════════════════════════════════════════════════════════════════════════
def render_yearly_distribution(df: pd.DataFrame) -> None:
    section_header(
        "Distribusi Tahunan",
        "Perbandingan jumlah dan proporsi ulasan per tahun untuk membaca perubahan persepsi pengguna secara adil antar periode.",
    )
    if not df["year"].notna().any():
        st.info("Kolom tahun tidak tersedia")
        return

    yearly = (
        df.dropna(subset=["year"]).groupby(["year", "label_name"], as_index=False)
        .size().rename(columns={"size": "count"})
    )
    yearly["year"] = yearly["year"].astype(int)
    yearly = yearly[yearly["label_name"].isin(SENTIMENT_ORDER)].copy()
    yearly["label_name"] = yearly["label_name"].astype(str)
    yearly["count"] = yearly["count"].astype(int)
    yearly["count_lbl"] = yearly["count"].map(lambda v: f"{v:,}")

    left, right = st.columns(2)
    with left:
        panel_label("Jumlah Ulasan per Tahun")
        fig_count = px.bar(
            yearly, x="year", y="count", color="label_name", barmode="group",
            category_orders={"label_name": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS, text="count_lbl",
        )
        fig_count.update_traces(textposition="outside", cliponaxis=False, texttemplate="%{text}")
        fig_count.update_layout(xaxis_title="Tahun", yaxis_title="Jumlah Ulasan")
        st.plotly_chart(polish(fig_count, 400, show_legend=True), use_container_width=True)

    with right:
        panel_label("Proporsi Sentimen per Tahun (%)")
        yearly_pct = yearly.copy()
        yearly_pct["pct"] = yearly_pct.groupby("year")["count"].transform(lambda s: s / s.sum() * 100).astype(float)
        yearly_pct["pct_lbl"] = yearly_pct["pct"].map(lambda v: f"{v:.1f}%")
        fig_pct = px.bar(
            yearly_pct, x="year", y="pct", color="label_name", barmode="stack",
            category_orders={"label_name": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS, text="pct_lbl",
        )
        fig_pct.update_traces(texttemplate="%{text}")
        fig_pct.update_layout(xaxis_title="Tahun", yaxis_title="Persentase (%)")
        st.plotly_chart(polish(fig_pct, 400, show_legend=True), use_container_width=True)

    if "score" in df.columns and df["score"].notna().any():
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        section_header(
            "Korelasi Rating vs Sentimen",
            "Seberapa konsisten rating bintang pengguna dengan sentimen teks ulasan yang diprediksi model.",
        )
        rating_df = (
            df.dropna(subset=["score"]).assign(score=lambda d: d["score"].astype(int))
            .groupby(["score", "label_name"], as_index=False)
            .size().rename(columns={"size": "count"})
        )
        rating_df = rating_df[rating_df["label_name"].isin(SENTIMENT_ORDER)].copy()
        rating_df["score"] = rating_df["score"].astype(int)
        rating_df["label_name"] = rating_df["label_name"].astype(str)
        rating_df["count"] = rating_df["count"].astype(int)
        rating_df["count_lbl"] = rating_df["count"].map(lambda v: f"{v:,}")
        fig_rating = px.bar(
            rating_df, x="score", y="count", color="label_name", barmode="group",
            category_orders={"label_name": SENTIMENT_ORDER},
            color_discrete_map=SENTIMENT_COLORS, text="count_lbl",
            labels={"score": "Rating Bintang (1–5)", "count": "Jumlah Ulasan"},
        )
        fig_rating.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
        fig_rating.update_layout(xaxis={"tickmode": "linear", "tick0": 1, "dtick": 1}, xaxis_title="Rating Bintang", yaxis_title="Jumlah Ulasan")
        st.plotly_chart(polish(fig_rating, 380, show_legend=True), use_container_width=True)
        st.caption("Rating 1–2 umumnya berkorelasi dengan sentimen negatif; rating 4–5 dengan sentimen positif.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUASI MODEL
# ════════════════════════════════════════════════════════════════════════════
def render_evaluation(summary: dict) -> None:
    section_header(
        "Evaluasi Model",
        "Performa model IndoBERT pada konfigurasi final 2-kelas — seberapa baik model membedakan ulasan positif dan negatif.",
    )
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

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    cm_payload = summary.get("confusionMatrix", {})
    if cm_payload:
        cm_df = pd.DataFrame(cm_payload).T.fillna(0)
        for col in cm_df.columns:
            cm_df[col] = pd.to_numeric(cm_df[col], errors="coerce").fillna(0).astype(int)

        row_totals = cm_df.values.sum(axis=1, keepdims=True)
        safe_totals = np.where(row_totals == 0, 1, row_totals)
        pct_matrix = cm_df.values / safe_totals * 100
        text_labels = [
            [f"{v}<br>({p:.1f}%)" for v, p in zip(row_v, row_p)]
            for row_v, row_p in zip(cm_df.values, pct_matrix)
        ]

        left, right = st.columns([1.28, 0.82])
        with left:
            panel_label("Confusion Matrix")
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_df.values.tolist(), x=list(cm_df.columns), y=list(cm_df.index),
                    colorscale=[[0, "#eef2ff"], [0.5, "#a5b4fc"], [1, "#4f46e5"]],
                    text=text_labels, texttemplate="%{text}",
                    textfont={"size": 14}, showscale=True,
                    hovertemplate="Aktual %{y} · Prediksi %{x}<br>Jumlah: %{z}<extra></extra>",
                )
            )
            fig_cm.update_layout(xaxis_title="Prediksi Model", yaxis_title="Label Aktual")
            st.plotly_chart(polish(fig_cm, 390, show_legend=False), use_container_width=True)
        with right:
            st.markdown(
                """
                <div class='note-box'>
                    <strong>Interpretasi cepat.</strong><br><br>
                    Model final sudah kuat secara umum, terlihat dari weighted F1 yang tinggi.
                    Namun performa antar kelas belum sepenuhnya seimbang: kelas negatif lebih
                    mudah dikenali dibanding kelas positif. Artinya, pembacaan insight positif
                    tetap perlu disertai kehati-hatian karena sebagian ulasan positif masih
                    dapat bergeser ke prediksi negatif.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class='note-box'>
            <strong>Catatan metodologis.</strong> Konfigurasi final memakai 2 kelas karena
            distribusi data tidak seimbang. Skema 3 kelas diposisikan sebagai eksperimen
            pembanding, bukan model final.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA PREDIKSI
# ════════════════════════════════════════════════════════════════════════════
def render_custom_table(df: pd.DataFrame, rows: int, sort_by: str) -> None:
    view = df.copy()
    if sort_by == "Terbaru" and "at" in view.columns:
        view = view.sort_values("at", ascending=False)
    elif sort_by == "Tertua" and "at" in view.columns:
        view = view.sort_values("at", ascending=True)
    elif sort_by == "Confidence ↑" and "confidence" in view.columns:
        view = view.sort_values("confidence", ascending=False)
    elif sort_by == "Confidence ↓" and "confidence" in view.columns:
        view = view.sort_values("confidence", ascending=True)

    cols = [c for c in ["at", "content", "score", "label_name", "confidence"] if c in view.columns]
    raw = view[cols].head(rows).copy()

    headers = {"at": "Tanggal", "content": "Ulasan", "score": "Rating", "label_name": "Sentimen", "confidence": "Confidence"}
    thead = "".join(f"<th>{html.escape(headers.get(col, col))}</th>" for col in cols)

    rows_html = []
    for _, row in raw.iterrows():
        cells = []
        for col in cols:
            raw_val = row[col]
            if col == "label_name":
                value = "" if pd.isna(raw_val) else str(raw_val)
                css = "pill-positive" if value.lower() == "positif" else "pill-negative"
                cells.append(f"<td><span class='sentiment-pill {css}'>{html.escape(value)}</span></td>")
            elif col == "content":
                value = "" if pd.isna(raw_val) else str(raw_val)
                cells.append(f"<td class='td-content'>{html.escape(value)}</td>")
            elif col == "at":
                value = "" if pd.isna(raw_val) else raw_val.strftime("%Y-%m-%d %H:%M")
                cells.append(f"<td class='td-date'>{html.escape(value)}</td>")
            elif col == "score":
                if pd.isna(raw_val):
                    value = ""
                else:
                    value = str(int(raw_val)) if float(raw_val).is_integer() else f"{raw_val}"
                cells.append(f"<td class='td-score'>{html.escape(value)} ★</td>" if value else "<td class='td-score'></td>")
            elif col == "confidence":
                if pd.isna(raw_val):
                    cells.append("<td class='td-confidence'>N/A</td>")
                else:
                    pct = float(raw_val) * 100
                    cells.append(
                        f"<td class='td-confidence'><span class='conf-mini'>"
                        f"<span class='conf-val'>{raw_val:.3f}</span>"
                        f"<span class='conf-track'><span class='conf-bar' style='width:{pct:.0f}%;'></span></span>"
                        f"</span></td>"
                    )
            else:
                value = "" if pd.isna(raw_val) else str(raw_val)
                cells.append(f"<td>{html.escape(value)}</td>")
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
    section_header(
        "Data Prediksi",
        "Periksa ulasan individual, label hasil prediksi, dan confidence model pada level data.",
    )

    col_sort, col_rows = st.columns([2, 1])
    with col_sort:
        sort_by = st.segmented_control(
            "Urutkan",
            options=["Terbaru", "Tertua", "Confidence ↑", "Confidence ↓"],
            default="Terbaru", key="table_sort",
        ) or "Terbaru"
    with col_rows:
        rows = st.segmented_control(
            "Baris", options=[10, 25, 50, 100], default=25, key="table_rows",
        ) or 25

    render_custom_table(df, rows, sort_by)
    st.caption(f"Menampilkan {min(rows, len(df)):,} dari {len(df):,} baris dengan filter aktif.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
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

    # top bar (rendered after load so we can show live stats)
    range_txt = "—"
    if df_all["year"].notna().any():
        ys = df_all["year"].dropna().astype(int)
        range_txt = f"{ys.min()}–{ys.max()}"
    st.markdown(
        f"""
        <div class='topbar'>
            <div class='topbar-left'>
                <div class='topbar-logo'>SA</div>
                <div>
                    <div class='topbar-title'>Sentiment Analytics — MyPertamina</div>
                    <div class='topbar-sub'>Klasifikasi sentimen ulasan pengguna aplikasi MyPertamina menggunakan IndoBERT</div>
                </div>
            </div>
            <div class='topbar-right'>
                <span class='chip'><span class='chip-dot'></span>Model aktif</span>
                <span class='chip'>IndoBERT · 2 Kelas</span>
                <span class='chip'>{len(df_all):,} ulasan</span>
                <span class='chip'>{range_txt}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_filtered.empty:
        st.warning("Tidak ada data sesuai filter yang dipilih.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Distribusi & Korelasi", "Evaluasi Model", "Data Prediksi",
    ])
    with tab1:
        render_overview(df_filtered, summary)
    with tab2:
        render_yearly_distribution(df_filtered)
    with tab3:
        render_evaluation(summary)
    with tab4:
        render_table(df_filtered)


if __name__ == "__main__":
    main()
