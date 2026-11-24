"""MusicMotion Studio — interactive music-driven character animation (CAT404).

Streamlit front end for the pipeline in ``src/``:

    audio -> features -> mapping plan -> motion -> Blender render -> evaluation

The interface is deliberately *rule-transparent*: every mapping weight is
visible and editable, because inspecting and adjusting the mapping is what
separates this system from a black-box dance generator.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import beat_alignment  # noqa: E402
import mapping  # noqa: E402
import motion_blend  # noqa: E402
import music_features  # noqa: E402
import paths  # noqa: E402
import pipeline  # noqa: E402

paths.ensure_dirs()

st.set_page_config(
    page_title="MusicMotion Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = PROJECT_DIR / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DIR = PROJECT_DIR / "outputs" / "user_study"
STUDY_DIR.mkdir(parents=True, exist_ok=True)

ACCENT = "#6C5CE7"
MUTED = "#6A7085"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:ital,opsz,wght@0,9..144,600;1,9..144,500&display=swap');

:root {
    --ink: #101426;
    --muted: #6A7085;
    --line: #E6E3DA;
    --paper: #F7F6F2;
    --accent: #6C5CE7;
    --accent-2: #00B8D9;
    --gold: #C89B52;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Manrope', -apple-system, 'Segoe UI', system-ui, sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 520px at 4% -10%, rgba(108, 92, 231, 0.10), transparent 62%),
        radial-gradient(900px 480px at 100% 0%, rgba(0, 184, 217, 0.08), transparent 58%),
        radial-gradient(700px 400px at 60% 110%, rgba(200, 155, 82, 0.07), transparent 60%),
        var(--paper);
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1080px;
    padding-top: 1.5rem;
    padding-bottom: 4.5rem;
}

h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.025em; }
h2 { font-weight: 750; margin-top: 0.5rem; }
h3 { font-weight: 700; font-size: 1.05rem; }

/* ---------------- sidebar ---------------- */
section[data-testid="stSidebar"] {
    background-color: #0C0F1E;
    background:
        radial-gradient(500px 300px at 20% -8%, rgba(108, 92, 231, 0.30), transparent 65%),
        radial-gradient(420px 260px at 110% 30%, rgba(0, 184, 217, 0.16), transparent 60%),
        linear-gradient(180deg, #0C0F1E 0%, #0A0C18 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Streamlit hides the sidebar below 800 px; pin it and shift the main column. */
@media (max-width: 820px) {
    section[data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        bottom: 0 !important;
        transform: none !important;
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
        z-index: 999;
        overflow-y: auto !important;
    }
    section[data-testid="stSidebar"] > div { width: 250px !important; }
    [data-testid="stMain"] {
        margin-left: 250px !important;
        width: calc(100% - 250px) !important;
    }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
    .hero h1 { font-size: 1.9rem; }
    .eq { display: none; }
}

/* Streamlit remembers a collapsed sidebar between reloads. Keep it open and
   remove the collapse toggle so the controls cannot disappear again. */
section[data-testid="stSidebar"][data-collapsed="true"] {
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
    flex: 0 0 300px !important;
    transform: none !important;
    margin-left: 0 !important;
    visibility: visible !important;
    display: block !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"][data-collapsed="true"] > div {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    width: 300px !important;
}
section[data-testid="stSidebar"][data-collapsed="true"] ~ [data-testid="stMain"] {
    margin-left: 300px !important;
}
button[aria-label="Collapse sidebar"] { display: none !important; }
@media (max-width: 820px) {
    section[data-testid="stSidebar"][data-collapsed="true"] { width: 250px !important; min-width: 250px !important; max-width: 250px !important; }
    section[data-testid="stSidebar"][data-collapsed="true"] > div { width: 250px !important; }
    section[data-testid="stSidebar"][data-collapsed="true"] ~ [data-testid="stMain"] { margin-left: 250px !important; }
}
section[data-testid="stSidebar"] * { color: #EDEEF7; }
section[data-testid="stSidebar"] .stSelectbox div[role="group"] { background: transparent !important; }
section[data-testid="stSidebar"] .stSelectbox input[role="combobox"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 11px !important;
    color: #EDEEF7 !important;
    padding: 0.5rem 0.7rem !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSelectbox button {
    background: transparent !important; color: #B9BFD8 !important; border: none !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #E9ECF8 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background-image: linear-gradient(135deg, #6C5CE7, #8B7CF6) !important;
    background-color: #6C5CE7 !important;
    color: #FFFFFF !important; border: none !important; border-radius: 9px !important;
    font-weight: 650 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(255,255,255,0.22) !important;
    border-radius: 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] section,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] div {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p { color: #AEB4CE !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: #AEB4CE !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.10); }
section[data-testid="stSidebar"] .stCheckbox label p { color: #E6E8F5 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #B9BFD8 !important; }

.brand { display: flex; align-items: center; gap: 0.75rem; padding: 0.1rem 0 1.15rem 0; }
.brand-mark {
    width: 44px; height: 44px; border-radius: 14px; position: relative;
    display: grid; place-items: center; font-size: 1.2rem; color: #fff;
    background: linear-gradient(145deg, #7C6BF0, #00B8D9);
    box-shadow: 0 10px 24px rgba(108, 92, 231, 0.45), inset 0 1px 0 rgba(255,255,255,0.35);
}
.brand-name { font-weight: 800; font-size: 1.02rem; letter-spacing: -0.02em; line-height: 1.15; }
.brand-sub { font-size: 0.68rem; color: #9AA1BE; letter-spacing: 0.16em; text-transform: uppercase; margin-top: 2px; }

.side-label {
    font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #8C93B0; font-weight: 700; margin: 0.2rem 0 0.35rem 0;
}

/* ---------------- hero ---------------- */
.hero {
    position: relative; overflow: hidden; border-radius: 24px;
    padding: 1.85rem 2rem 1.65rem 2rem;
    background:
        radial-gradient(680px 260px at 96% -10%, rgba(0, 184, 217, 0.35), transparent 60%),
        radial-gradient(520px 280px at 0% 110%, rgba(200, 155, 82, 0.20), transparent 60%),
        linear-gradient(118deg, #0E1122 0%, #191E3D 52%, #3A2E86 100%);
    color: #FFFFFF;
    box-shadow: 0 30px 70px -24px rgba(16, 20, 38, 0.55);
}
.hero:before {
    content: ""; position: absolute; inset: 0; pointer-events: none; opacity: 0.15;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    mix-blend-mode: overlay;
}
.hero:after {
    content: ""; position: absolute; right: -90px; bottom: -130px; width: 320px; height: 320px;
    border-radius: 50%; border: 1px solid rgba(255,255,255,0.12);
    background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.12), transparent 62%);
}
.hero-grid { position: relative; z-index: 2; display: flex; justify-content: space-between; gap: 2rem; }
.kicker {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-size: 0.68rem; letter-spacing: 0.22em; text-transform: uppercase;
    color: #C7C2FF; font-weight: 700; margin-bottom: 0.55rem;
}
.kicker .dot { width: 6px; height: 6px; border-radius: 50%; background: #00E0C6; box-shadow: 0 0 12px #00E0C6; }
.hero h1 {
    color: #FFFFFF; font-size: 2.55rem; line-height: 1.05; margin: 0 0 0.5rem 0;
    font-weight: 800; letter-spacing: -0.035em;
}
.hero h1 em { font-family: 'Fraunces', Georgia, serif; font-style: italic; font-weight: 500; color: #D9D4FF; }
.hero p { color: rgba(228, 230, 250, 0.82); font-size: 0.95rem; line-height: 1.55; max-width: 640px; margin: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 1.1rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.34rem 0.75rem; border-radius: 999px; font-size: 0.74rem; font-weight: 650;
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14);
    color: #F4F4FF; backdrop-filter: blur(6px);
}
.chip svg { width: 12px; height: 12px; opacity: 0.85; }

.eq { display: flex; align-items: flex-end; gap: 5px; height: 74px; margin-top: 0.4rem; }
.eq span {
    width: 7px; border-radius: 4px; background: linear-gradient(180deg, #00E0C6, #6C5CE7);
    opacity: 0.9; animation: pulse 1.1s ease-in-out infinite;
}
.eq span:nth-child(1) { height: 26px; animation-delay: 0s; }
.eq span:nth-child(2) { height: 52px; animation-delay: .12s; }
.eq span:nth-child(3) { height: 38px; animation-delay: .24s; }
.eq span:nth-child(4) { height: 66px; animation-delay: .36s; }
.eq span:nth-child(5) { height: 44px; animation-delay: .48s; }
.eq span:nth-child(6) { height: 58px; animation-delay: .60s; }
.eq span:nth-child(7) { height: 30px; animation-delay: .72s; }
@keyframes pulse { 0%, 100% { transform: scaleY(0.55); } 50% { transform: scaleY(1); } }

/* ---------------- cards ---------------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid transparent !important;
    border-radius: 18px !important;
    background:
        linear-gradient(#FFFFFF, #FFFFFF) padding-box,
        linear-gradient(135deg, rgba(108,92,231,0.30), rgba(0,184,217,0.16) 45%, rgba(200,155,82,0.22)) border-box !important;
    box-shadow: 0 1px 2px rgba(16,20,38,0.05), 0 18px 44px -20px rgba(16,20,38,0.30) !important;
    padding: 1.15rem 1.2rem !important;
}

.sec { display: flex; align-items: center; gap: 0.7rem; margin: 0.1rem 0 0.1rem 0; }
.sec-ico {
    width: 32px; height: 32px; border-radius: 10px; display: grid; place-items: center;
    background: linear-gradient(145deg, #F1EEFF, #E7F8FF);
    border: 1px solid #E4E0FF; font-size: 0.95rem;
}
.sec-t { font-size: 1.02rem; font-weight: 750; letter-spacing: -0.015em; }
.sec-h { font-size: 0.76rem; color: var(--muted); margin-top: 1px; }
.rule { height: 3px; width: 54px; border-radius: 99px; margin: 0.5rem 0 0.9rem 0;
    background: linear-gradient(90deg, #6C5CE7, #00B8D9 60%, transparent); }

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 0.9rem 1rem 0.85rem 1rem;
    box-shadow: 0 12px 30px -22px rgba(16,20,38,0.45);
    position: relative; overflow: hidden;
}
[data-testid="stMetric"]:before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #6C5CE7, #00B8D9);
}
[data-testid="stMetricLabel"] p {
    color: var(--muted); font-size: 0.68rem; font-weight: 750;
    letter-spacing: 0.14em; text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--ink); font-weight: 800; font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}

/* ---------------- tabs ---------------- */
/* ---------------- tabs (Streamlit 1.66 react-aria markup) ---------------- */
.stTabs [role="tablist"] {
    gap: 0.3rem; background: rgba(255,255,255,0.72);
    padding: 0.32rem; border-radius: 16px; border: 1px solid var(--line);
    box-shadow: 0 10px 30px -22px rgba(16,20,38,0.6);
    backdrop-filter: blur(8px);
}
.stTabs [data-testid="stTab"] {
    height: 44px; border-radius: 12px; padding: 0 1.1rem;
    display: flex; align-items: center; justify-content: center;
    font-weight: 650; color: #5B6178; background: transparent;
    transition: all 0.18s ease;
}
.stTabs [data-testid="stTab"] p {
    font-size: 0.88rem; font-weight: 650; color: inherit; margin: 0;
}
.stTabs [data-testid="stTab"]:hover { color: #3A2E86; background: rgba(108,92,231,0.06); }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    background: linear-gradient(135deg, #6C5CE7, #7E6CF2);
    box-shadow: 0 8px 20px -8px rgba(108,92,231,0.75), inset 0 1px 0 rgba(255,255,255,0.25);
}
.stTabs [data-testid="stTab"][aria-selected="true"] p { color: #FFFFFF !important; }
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stAppDeployButton"], #MainMenu { display: none !important; }

/* ---------------- controls ---------------- */
.stButton > button, [data-testid="stDownloadButton"] > button {
    border-radius: 11px; font-weight: 650; font-size: 0.88rem; padding: 0.5rem 1.05rem;
    border: 1px solid var(--line); background: #FFFFFF; color: var(--ink);
    transition: transform 0.12s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    border-color: #C9C2F5; color: #3A2E86; transform: translateY(-1px);
    box-shadow: 0 12px 26px -14px rgba(108,92,231,0.55);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7, #8B7CF6);
    color: #FFFFFF; border: none;
    box-shadow: 0 14px 30px -12px rgba(108,92,231,0.75), inset 0 1px 0 rgba(255,255,255,0.28);
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 18px 36px -12px rgba(108,92,231,0.85); }

[data-testid="stExpander"] details {
    border: 1px solid var(--line); border-radius: 16px; background: #FFFFFF;
    box-shadow: 0 12px 30px -24px rgba(16,20,38,0.5);
}
[data-testid="stExpander"] summary { font-weight: 650; }

[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden; border: 1px solid var(--line);
    background: #FFFFFF;
}
[data-testid="stDataFrame"] * { font-family: 'Manrope', system-ui, sans-serif; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, #6C5CE7, #00B8D9); }

.stAudio audio { width: 100%; border-radius: 12px; }

.empty {
    border: 1px dashed #CFCBE0; border-radius: 18px; padding: 1.9rem 1.4rem;
    text-align: center; color: var(--muted);
    background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(250,249,246,0.65));
}
.empty .ring {
    width: 54px; height: 54px; margin: 0 auto 0.7rem auto; border-radius: 50%;
    display: grid; place-items: center; font-size: 1.35rem;
    background: radial-gradient(circle at 30% 30%, #FFFFFF, #F0EDFF);
    border: 1px solid #E2DDFF; box-shadow: 0 10px 24px -14px rgba(108,92,231,0.6);
}
.empty .big { font-weight: 750; color: var(--ink); margin-bottom: 0.25rem; font-size: 0.98rem; }
.empty .note { font-size: 0.83rem; line-height: 1.5; }

.note { font-size: 0.83rem; color: var(--muted); line-height: 1.55; }
.pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.24rem 0.62rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 750; letter-spacing: 0.02em;
    border: 1px solid transparent;
}
.pill-ok { background: #E8F8F0; color: #12805A; border-color: #C6EEDB; }
.pill-warn { background: #FFF4E4; color: #A96513; border-color: #F3DFBE; }
.pill-info { background: #EEECFF; color: #4B3FA8; border-color: #DCD7FF; }

.player { display: flex; align-items: center; gap: 0.9rem; }
.disc {
    width: 46px; height: 46px; border-radius: 50%; flex: 0 0 auto;
    background:
        radial-gradient(circle at 50% 50%, #0E1122 0 17%, transparent 17.5%),
        repeating-radial-gradient(circle at 50% 50%, rgba(255,255,255,0.10) 0 1px, transparent 1px 4px),
        conic-gradient(from 0deg, #1A1F3D, #2C2F58, #1A1F3D, #3A2E86, #1A1F3D);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12), 0 10px 22px -12px rgba(16,20,38,0.8);
    animation: spin 14s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

footer { visibility: hidden; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

ICONS = {
    "music": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'><path d='M9 18V5l12-2v13'/><circle cx='6' cy='18' r='3'/><circle cx='18' cy='16' r='3'/></svg>",
    "layers": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'><path d='M12 2 2 7l10 5 10-5-10-5Z'/><path d='m2 17 10 5 10-5'/><path d='m2 12 10 5 10-5'/></svg>",
    "play": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'><path d='m6 3 14 9-14 9V3Z'/></svg>",
    "chart": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'><path d='M3 3v18h18'/><path d='m7 14 4-4 3 3 5-6'/></svg>",
    "book": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'><path d='M4 19.5A2.5 2.5 0 0 1 6.5 17H20'/><path d='M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z'/></svg>",
    "wave": "<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8'><path d='M2 12h3l2-7 3 14 3-10 2 5h5'/></svg>",
}


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def state(key: str, default=None):
    return st.session_state.get(key, default)


def reset_results() -> None:
    for key in ("plan", "motion_dir", "metrics", "analysis", "render_path", "run_name", "preview"):
        st.session_state.pop(key, None)


def analyse(path: Path, refresh: bool = False):
    key = ("analysis", str(path), refresh)
    if state("analysis_key") == key and state("analysis") is not None:
        return state("analysis")

    with st.status("Analysing the music…", expanded=False) as status:
        analysis = music_features.analyse_music(path, out_dir=paths.feature_dir(path), make_plot=True)
        status.update(label="Analysis complete", state="complete")

    st.session_state["analysis"] = analysis
    st.session_state["analysis_key"] = key
    return analysis


def make_plan(analysis, rules, mode):
    library = pd.read_csv(paths.LIBRARY_INDEX_PATH)
    return mapping.build_plan(
        beats=analysis.beats,
        library=library,
        rules=rules,
        mode=mode,
        duration=analysis.duration,
    )


def build_motion_and_evaluate(analysis, plan, rules, mode, run_name):
    run_dir = paths.run_dir(run_name)
    plan.save(run_dir)

    with st.status("Assembling and blending motion…", expanded=False) as status:
        library = motion_blend.ClipLibrary()
        motion = motion_blend.build_motion(plan.plan, library, rules)
        motion.save(run_dir)
        status.update(label="Motion assembled", state="complete")

    with st.status("Computing beat-alignment metrics…", expanded=False) as status:
        metrics = beat_alignment.evaluate_run(
            run_dir, feature_dir=paths.feature_dir(Path(analysis.audio_path))
        )
        status.update(label="Metrics ready", state="complete")

    st.session_state["motion_dir"] = run_dir
    st.session_state["metrics"] = metrics
    st.session_state["run_name"] = run_name
    st.session_state["render_path"] = None
    return run_dir, metrics


def preview_image(run_dir: Path) -> Path | None:
    script = PROJECT_DIR / "src" / "preview_motion.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run", str(run_dir)],
        capture_output=True, text=True, errors="replace",
        cwd=str(PROJECT_DIR), timeout=300,
    )
    image = run_dir / "motion_preview.png"
    return image if result.returncode == 0 and image.exists() else None


def render_video(run_dir: Path, audio_path: Path, mode: str) -> Path | None:
    script = PROJECT_DIR / "blender" / "render_character.py"
    output = run_dir / f"character_{mode}.mp4"
    command = [
        str(paths.BLENDER_EXE), "--background", "--factory-startup",
        "--python", str(script), "--",
        "--motion", str(run_dir / "motion.npz"),
        "--audio", str(audio_path),
        "--output", str(output),
    ]
    started = time.time()
    with st.status("Blender is rendering the character…", expanded=True) as status:
        st.write("A 20-second clip takes roughly 8–15 minutes on a laptop.")
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            cwd=str(PROJECT_DIR), timeout=3600,
        )
        status.update(label=f"Render finished in {time.time() - started:.0f} s", state="complete")

    (run_dir / "render.log").write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0 or not output.exists():
        st.error("Blender render failed. See render.log for details.")
        st.code((result.stderr or "")[-3000:])
        return None
    return output


RUNS_DIR = PROJECT_DIR / "outputs" / "runs"
EXPERIMENTS_DIR = PROJECT_DIR / "outputs" / "experiments"


@st.cache_data(ttl=30, show_spinner=False)
def load_runs() -> pd.DataFrame:
    """One row per recorded run (newest first)."""
    rows = []
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            mode = ""
            summary_path = run_dir / "mapping_summary.json"
            if summary_path.exists():
                try:
                    mode = json.loads(summary_path.read_text(encoding="utf-8")).get("mode", "")
                except Exception:
                    mode = ""
            rows.append({
                "run": run_dir.name,
                "mode": mode or metrics.get("mode", ""),
                "bas": metrics.get("bas"),
                "beat_f1": metrics.get("beat_f1"),
                "energy_corr": metrics.get("energy_corr"),
                "clip_variety": metrics.get("clip_variety"),
                "slots": metrics.get("slots"),
                "video": any(run_dir.glob("character_*.mp4")),
                "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def library_frame() -> pd.DataFrame:
    return pd.read_csv(paths.LIBRARY_INDEX_PATH)


@st.cache_data(ttl=60, show_spinner=False)
def experiment_summary() -> pd.DataFrame:
    path = EXPERIMENTS_DIR / "summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-grid">
            <div>
              <div class="kicker"><span class="dot"></span>CAT404 · Final Year Project · build v3</div>
              <h1>MusicMotion <em>Studio</em></h1>
              <p>An interpretable music-driven 3D character animation system. Beat, tempo, energy,
              onset and structure are mapped onto motion by explicit, editable rules — then rendered
              in Blender and measured with Beat Alignment Score.</p>
              <div class="chips">
                <span class="chip">{ICONS['layers']} Multi-level mapping</span>
                <span class="chip">{ICONS['wave']} 16 737 motion clips</span>
                <span class="chip">{ICONS['play']} Blender rendering</span>
                <span class="chip">{ICONS['chart']} BAS evaluation</span>
              </div>
            </div>
            <div style="display:flex;align-items:flex-end">
              <div class="eq"><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(icon_key: str, title: str, hint: str = "") -> None:
    hint_html = f"<div class='sec-h'>{hint}</div>" if hint else ""
    st.markdown(
        f"""
        <div class="sec">
          <div class="sec-ico">{ICONS.get(icon_key, '')}</div>
          <div><div class="sec-t">{title}</div>{hint_html}</div>
        </div>
        <div class="rule"></div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="empty">
          <div class="ring">{icon}</div>
          <div class="big">{title}</div>
          <div class="note">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info") -> str:
    return f"<span class='pill pill-{kind}'>{text}</span>"


def altair_theme(chart, height: int = 240):
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(grid=False, domain=False, ticks=False,
                        labelColor=MUTED, labelFontSize=10, titleColor=MUTED)
        .configure(background="transparent")
    )


def rules_from_sidebar(base: dict, mode: str) -> dict:
    rules = json.loads(json.dumps(base))
    rules["mode"] = mode

    st.sidebar.markdown("<div class='side-label'>Mapping weights</div>", unsafe_allow_html=True)
    st.sidebar.caption("Every value is recorded with the run.")

    rules["slot_beats"] = st.sidebar.slider("Beats per motion slot", 1, 4, int(base.get("slot_beats", 2)))

    with st.sidebar.expander("Tempo layer", expanded=False):
        rules["tempo"]["enabled"] = st.checkbox("Active", value=bool(base["tempo"]["enabled"]), key="tempo_on")
        rules["tempo"]["ref_bpm"] = float(st.slider("Reference tempo (BPM)", 80.0, 160.0, float(base["tempo"]["ref_bpm"]), 1.0, key="ref_bpm"))
        rules["tempo"]["weight"] = float(st.slider("Retrieval weight", 0.0, 1.0, float(base["tempo"]["weight"]), 0.05, key="tempo_w"))

    with st.sidebar.expander("Energy layer", expanded=False):
        rules["energy"]["enabled"] = st.checkbox("Active", value=bool(base["energy"]["enabled"]), key="energy_on")
        rules["energy"]["min_gain"] = float(st.slider("Minimum gain", 0.3, 1.0, float(base["energy"]["min_gain"]), 0.05, key="min_gain"))
        rules["energy"]["max_gain"] = float(st.slider("Maximum gain", 1.0, 2.0, float(base["energy"]["max_gain"]), 0.05, key="max_gain"))
        rules["energy"]["weight"] = float(st.slider("Retrieval weight", 0.0, 1.0, float(base["energy"]["weight"]), 0.05, key="energy_w"))

    with st.sidebar.expander("Onset layer", expanded=False):
        rules["onset"]["enabled"] = st.checkbox("Active", value=bool(base["onset"]["enabled"]), key="onset_on")
        rules["onset"]["min_blend"] = float(st.slider("Blend for sharp onsets (s)", 0.05, 0.4, float(base["onset"]["min_blend"]), 0.01, key="min_blend"))
        rules["onset"]["max_blend"] = float(st.slider("Blend for soft onsets (s)", 0.1, 0.6, float(base["onset"]["max_blend"]), 0.01, key="max_blend"))

    with st.sidebar.expander("Structure and selection", expanded=False):
        rules["structure"]["enabled"] = st.checkbox("Active", value=bool(base["structure"]["enabled"]), key="struct_on")
        rules["selection"]["diversity_penalty"] = float(st.slider("Clip diversity penalty", 0.0, 0.6, float(base["selection"]["diversity_penalty"]), 0.05, key="diversity"))

    with st.sidebar.expander("Rendering and stage", expanded=False):
        rules["blend"]["root_anchor"] = float(st.slider("Stage anchoring", 0.0, 1.0, float(base["blend"].get("root_anchor", 0.45)), 0.05, key="anchor"))
        rules["blend"]["foot_lock"] = st.checkbox("Keep feet on the floor", value=bool(base["blend"].get("foot_lock", True)), key="foot_lock")

    return rules


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

BASE_RULES = paths.load_json(paths.RULES_PATH, {})
if not BASE_RULES:
    st.error("config/mapping_rules.json is missing.")
    st.stop()

if not paths.LIBRARY_INDEX_PATH.exists():
    st.error("Motion library missing. Run `python src/motion_library.py` once.")
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">♪</div>
            <div>
                <div class="brand-name">MusicMotion</div>
                <div class="brand-sub">Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='side-label'>Music input</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"], label_visibility="collapsed")
    demos = sorted(path.name for path in paths.audio_files())
    demo_choice = st.selectbox("or use a bundled song", demos, index=0 if demos else None)

    st.divider()
    st.markdown("<div class='side-label'>Mapping mode</div>", unsafe_allow_html=True)
    mode = st.selectbox(
        "Mapping mode",
        options=list(mapping.MODES),
        index=0,
        label_visibility="collapsed",
        format_func=lambda value: {
            "multi_level": "Multi-level · full system",
            "beat_only": "Beat-only · baseline",
            "no_tempo": "Ablation · no tempo",
            "no_energy": "Ablation · no energy",
            "no_onset": "Ablation · no onset",
            "no_structure": "Ablation · no structure",
        }[value],
    )
    st.caption("All conditions share the same library, blending and interface.")

    rules = rules_from_sidebar(BASE_RULES, mode)

audio_path: Path | None = None
if uploaded is not None:
    digest = hashlib.md5(uploaded.getvalue()).hexdigest()[:10]
    audio_path = UPLOAD_DIR / f"upload_{digest}.wav"
    if not audio_path.exists():
        audio_path.write_bytes(uploaded.getvalue())
elif demo_choice:
    audio_path = paths.AUDIO_DIR / demo_choice

hero()

if audio_path is None or not audio_path.exists():
    empty_state("🎧", "Choose a song to begin", "Upload a WAV file or pick a bundled demo song in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Song strip
# ---------------------------------------------------------------------------

st.write("")
with st.container(border=True):
    info_col, audio_col = st.columns([1.7, 1.0], gap="large")
    with info_col:
        st.markdown(
            f"""
            <div class="player">
              <div class="disc"></div>
              <div>
                <div class="note" style="text-transform:uppercase;letter-spacing:.16em;font-weight:750;color:#6A7085">
                    Now working on
                </div>
                <div style="font-size:1.28rem;font-weight:800;letter-spacing:-.02em;line-height:1.25">{audio_path.name}</div>
                <div style="margin-top:.35rem">{pill(mode, 'info')}&nbsp;{pill('rules editable', 'ok')}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with audio_col:
        st.audio(str(audio_path))

analysis = state("analysis")
if analysis is None or state("analysis_key", (None,))[1] != str(audio_path):
    analysis = analyse(audio_path)

tabs = st.tabs([
    "◎ Dashboard",
    "① Music analysis",
    "② Mapping plan",
    "③ Motion & render",
    "④ Evaluation",
    "⑤ Documentation",
])


# ---------------------------------------------------------------------------
# Tab 0 — dashboard
# ---------------------------------------------------------------------------

with tabs[0]:
    section("chart", "Studio dashboard", "library · runs · pipeline · experiments")

    runs = load_runs()
    library = library_frame()
    experiments = experiment_summary()

    best_bas = float(runs["bas"].max()) if not runs.empty else 0.0
    mean_energy = float(runs["energy_corr"].mean()) if not runs.empty else 0.0

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.metric("Motion library", f"{len(library):,}", help="two-second clips indexed from AIST++")
    k2.metric("Recorded runs", len(runs))
    k3.metric("Best BAS", f"{best_bas:.3f}")
    k4.metric("Mean energy corr.", f"{mean_energy:.3f}")

    st.write("")
    pipe_col, action_col = st.columns([1.35, 1.0], gap="large")

    with pipe_col:
        with st.container(border=True):
            section("layers", "Pipeline status", "this session · selected song")
            steps = [
                ("Analysis", analysis is not None, "features + sections"),
                ("Mapping plan", state("plan") is not None, "one clip per slot"),
                ("Motion", state("motion_dir") is not None, "blended animation"),
                ("Render", bool(state("render_path")), "Blender MP4"),
                ("Evaluation", not experiments.empty, "experiment table"),
            ]
            rows_html = "".join(
                "<div style='display:flex;justify-content:space-between;align-items:center;"
                "padding:.45rem 0;border-bottom:1px solid #F1EFE8'>"
                f"<div><span style='font-weight:750'>{name}</span>"
                f"<span class='note' style='margin-left:.5rem'>{hint}</span></div>"
                f"{pill('done' if ok else 'pending', 'ok' if ok else 'warn')}</div>"
                for name, ok, hint in steps
            )
            st.markdown(rows_html, unsafe_allow_html=True)

    with action_col:
        with st.container(border=True):
            section("play", "Quick actions", "runs in this session")
            if st.button("Build mapping plan", type="primary", width="stretch"):
                try:
                    st.session_state["plan"] = make_plan(analysis, rules, mode)
                    st.toast("Mapping plan ready", icon="✅")
                except Exception as error:
                    st.error(f"Mapping failed: {error}")
            if st.button("Generate motion", width="stretch"):
                try:
                    if state("plan") is None:
                        st.session_state["plan"] = make_plan(analysis, rules, mode)
                    run_name = f"{paths.slug(audio_path.stem)}_{mode}"
                    run_dir, metrics = build_motion_and_evaluate(
                        analysis, state("plan"), rules, mode, run_name
                    )
                    image = preview_image(run_dir)
                    st.session_state["preview"] = str(image) if image else None
                    st.toast("Motion generated", icon="✨")
                except Exception as error:
                    st.error(f"Motion generation failed: {error}")
            st.caption("Detailed controls are in the tabs below.")

    st.write("")
    runs_col, trend_col = st.columns([1.35, 1.0], gap="large")

    with runs_col:
        with st.container(border=True):
            section("layers", "Recent runs", f"{len(runs)} recorded")
            if runs.empty:
                empty_state("🗂", "No runs yet", "Build a plan and generate motion to record the first run.")
            else:
                st.dataframe(
                    runs.head(7)[["run", "mode", "bas", "beat_f1", "energy_corr", "video"]],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "run": st.column_config.TextColumn("run", width="medium"),
                        "mode": st.column_config.TextColumn("mode", width="small"),
                        "bas": st.column_config.NumberColumn("BAS", format="%.3f"),
                        "beat_f1": st.column_config.NumberColumn("F1", format="%.3f"),
                        "energy_corr": st.column_config.NumberColumn("energy", format="%.3f"),
                        "video": st.column_config.CheckboxColumn("MP4"),
                    },
                )

    with trend_col:
        with st.container(border=True):
            section("chart", "BAS across runs", "newest eight runs")
            if len(runs) < 2:
                st.caption("Record at least two runs to see the trend.")
            else:
                trend = runs.head(8).iloc[::-1].reset_index(drop=True)
                trend["step"] = range(1, len(trend) + 1)
                line = (
                    alt.Chart(trend)
                    .mark_line(point=alt.OverlayMarkDef(size=55, filled=True, color=ACCENT),
                               color=ACCENT, strokeWidth=2.2)
                    .encode(
                        x=alt.X("step:O", title=None),
                        y=alt.Y("bas:Q", title="BAS", scale=alt.Scale(zero=False)),
                        tooltip=["run", "mode", "bas", "energy_corr"],
                    )
                )
                st.altair_chart(altair_theme(line, 210), width="stretch")

    st.write("")
    with st.container(border=True):
        section("book", "Experiment snapshot", "four songs × six conditions")
        if experiments.empty:
            st.caption("No experiment table yet — run `python src/experiment.py`.")
        else:
            melted = experiments.melt(
                id_vars="mode",
                value_vars=["energy_corr_mean", "beat_response_corr_mean"],
                var_name="metric", value_name="value",
            )
            melted["metric"] = melted["metric"].map({
                "energy_corr_mean": "energy correlation",
                "beat_response_corr_mean": "beat-response correlation",
            })
            bars = (
                alt.Chart(melted)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("mode:N", title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset="metric:N",
                    y=alt.Y("value:Q", title=None),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(range=["#6C5CE7", "#00B8D9"]),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    tooltip=["mode", "metric", alt.Tooltip("value:Q", format=".3f")],
                )
            )
            st.altair_chart(altair_theme(bars, 230), width="stretch")
            best = experiments.sort_values("energy_corr_mean", ascending=False).iloc[0]
            st.caption(
                f"Highest energy correlation: {best['mode']} ({best['energy_corr_mean']:.3f}). "
                "Generated by src/experiment.py under identical pipeline settings."
            )


# ---------------------------------------------------------------------------
# Tab 1 — analysis
# ---------------------------------------------------------------------------

with tabs[1]:
    section("music", "Musical analysis", "beat · tempo · energy · onset · structure")

    head_l, head_r = st.columns([1.0, 0.32])
    with head_r:
        if st.button("Re-analyse", type="secondary", width="stretch"):
            reset_results()
            analyse(audio_path, refresh=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Duration", f"{analysis.duration:.1f} s")
    c2.metric("Tempo", f"{analysis.tempo_bpm:.1f} BPM")
    c3.metric("Detected beats", len(analysis.beats))
    c4.metric("Sections", len(analysis.sections))

    st.write("")
    with st.container(border=True):
        section("wave", "Feature overview", "waveform · RMS · onset · sections")
        plot_path = paths.feature_dir(audio_path) / "feature_plot.png"
        if plot_path.exists():
            st.image(str(plot_path), width="stretch")
            st.caption(
                "Waveform, RMS energy, onset strength, spectral centroid and detected sections. "
                "The section bands are the structural layer used by the mapping."
            )
        else:
            empty_state("🖼", "No feature plot yet", "Run the analysis to generate the overview.")

    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
        with st.container(border=True):
            section("layers", "Sections", "energy rank drives the motion family")
            if analysis.sections:
                st.dataframe(
                    pd.DataFrame(analysis.sections)[
                        ["label", "start_time", "end_time", "duration", "mean_energy", "energy_rank"]
                    ].round(3),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No section boundaries were detected in this track.")
    with right:
        with st.container(border=True):
            section("chart", "Beat features", "the table the mapping consumes")
            st.dataframe(
                analysis.beats[["beat_time", "bpm_local", "energy_norm", "onset_norm", "section_label"]].round(3),
                width="stretch",
                hide_index=True,
                height=300,
            )


# ---------------------------------------------------------------------------
# Tab 2 — plan
# ---------------------------------------------------------------------------

with tabs[2]:
    section("layers", "Music-to-motion mapping", "one clip decision per musical phrase")

    st.caption(
        "Each row is a phrase slot of two beats. The beat grid fixes when movement happens; tempo, "
        "energy, onset and section decide which clip is retrieved and how it is played."
    )

    if st.button("Build mapping plan", type="primary"):
        try:
            with st.spinner("Retrieving clips for every slot…"):
                st.session_state["plan"] = make_plan(analysis, rules, mode)
            st.toast("Mapping plan ready", icon="✅")
        except Exception as error:
            st.error(f"Mapping failed: {error}")

    plan = state("plan")
    if plan is None:
        empty_state("🧭", "No plan yet", "Click “Build mapping plan” to see every clip decision.")
    else:
        st.write("")
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        m1.metric("Slots", len(plan.plan))
        m2.metric("Mean gain", f"{plan.plan['gain'].mean():.2f}×")
        m3.metric("Mean blend", f"{plan.plan['blend_seconds'].mean():.2f} s")
        m4.metric("Unique clips", int(plan.plan["clip_id"].nunique()))

        st.write("")
        with st.container(border=True):
            section("layers", "Slot decisions", f"{len(plan.plan)} slots · mode {mode}")
            st.dataframe(
                plan.plan[[
                    "slot", "start_time", "end_time", "bpm", "rate", "gain",
                    "blend_seconds", "energy_target", "section_label", "clip_id",
                    "clip_genre", "clip_tempo_bpm",
                ]].round(3),
                width="stretch",
                hide_index=True,
                height=360,
                column_config={
                    "start_time": st.column_config.NumberColumn("start (s)", format="%.2f"),
                    "end_time": st.column_config.NumberColumn("end (s)", format="%.2f"),
                    "bpm": st.column_config.NumberColumn("BPM", format="%.1f"),
                    "rate": st.column_config.NumberColumn("rate", format="%.2f"),
                    "gain": st.column_config.NumberColumn("gain", format="%.2f"),
                    "blend_seconds": st.column_config.NumberColumn("blend (s)", format="%.2f"),
                    "energy_target": st.column_config.NumberColumn("energy", format="%.2f"),
                    "clip_tempo_bpm": st.column_config.NumberColumn("clip BPM", format="%.0f"),
                },
            )

        chart_col, info_col = st.columns([1.6, 1.0], gap="large")
        with chart_col:
            with st.container(border=True):
                section("chart", "Intensity over time", "gain and playback rate per slot")
                chart_df = plan.plan[["start_time", "gain", "rate"]].rename(
                    columns={"start_time": "time"}
                )
                base = alt.Chart(chart_df).encode(x=alt.X("time:Q", title=None))
                area = base.mark_area(
                    line={"color": ACCENT, "strokeWidth": 2.2},
                    color=alt.Gradient(
                        gradient="linear",
                        stops=[
                            alt.GradientStop(color="rgba(108,92,231,0.34)", offset=0),
                            alt.GradientStop(color="rgba(108,92,231,0.02)", offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    ),
                ).encode(y=alt.Y("gain:Q", title="gain", scale=alt.Scale(zero=False)))
                line = base.mark_line(color="#00B8D9", strokeWidth=1.8, strokeDash=[5, 3]).encode(
                    y=alt.Y("rate:Q", title="rate", scale=alt.Scale(zero=False))
                )
                chart = alt.layer(area, line).resolve_scale(y="independent")
                st.altair_chart(altair_theme(chart, 250), width="stretch")
        with info_col:
            with st.container(border=True):
                section("chart", "Plan summary", "stored with the run")
                genres = pd.Series(plan.summary.get("genre_distribution", {})).sort_values()
                if len(genres):
                    genre_df = genres.rename_axis("genre").reset_index(name="clips")
                    bars = (
                        alt.Chart(genre_df)
                        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5,
                                  color="#6C5CE7", size=14)
                        .encode(
                            x=alt.X("clips:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                            y=alt.Y("genre:N", title=None, sort="-x"),
                            tooltip=["genre", "clips"],
                        )
                    )
                    st.altair_chart(altair_theme(bars, 170), width="stretch")
                st.caption(
                    f"Mean rate {plan.summary['mean_rate']:.2f}× · "
                    f"mean gain {plan.summary['mean_gain']:.2f}× · "
                    f"{plan.summary['unique_sources']} source recordings"
                )

        with st.expander("Explain every decision", expanded=False):
            st.text("\n".join(plan.explanation))

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download plan CSV",
            plan.plan.to_csv(index=False).encode("utf-8"),
            file_name=f"plan_{mode}.csv",
            mime="text/csv",
            width="stretch",
        )
        d2.download_button(
            "Download mapping rules (JSON)",
            json.dumps(plan.rules, indent=2).encode("utf-8"),
            file_name=f"rules_{mode}.json",
            mime="application/json",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Tab 3 — motion and render
# ---------------------------------------------------------------------------

with tabs[3]:
    section("play", "Motion generation", "Slerp blending · DTW alignment · ground correction")

    col_a, col_b = st.columns([1.0, 1.0], gap="large")
    with col_a:
        if st.button("Generate motion", type="primary", width="stretch"):
            if state("plan") is None:
                st.session_state["plan"] = make_plan(analysis, rules, mode)
            try:
                run_name = f"{paths.slug(audio_path.stem)}_{mode}"
                run_dir, metrics = build_motion_and_evaluate(
                    analysis, state("plan"), rules, mode, run_name
                )
                image = preview_image(run_dir)
                st.session_state["preview"] = str(image) if image else None
                st.toast("Motion generated", icon="✨")
            except Exception as error:
                st.exception(error)
    with col_b:
        st.markdown(
            "<div class='note'>Clips are time-warped, scaled by the energy gain and blended with "
            "spherical interpolation; a ground pass keeps the feet on the floor.</div>",
            unsafe_allow_html=True,
        )

    motion_dir = state("motion_dir")
    if motion_dir is None:
        empty_state("🕺", "No motion yet", "Build a plan, then generate the animation.")
    else:
        metrics = state("metrics") or {}
        st.write("")
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        k1.metric("BAS", f"{metrics.get('bas', 0):.3f}", help="Gaussian beat alignment score")
        k2.metric("Beat F1", f"{metrics.get('beat_f1', 0):.3f}")
        k3.metric("Energy corr.", f"{metrics.get('energy_corr', 0):.3f}")
        k4.metric("Dynamic range", f"{metrics.get('dynamic_range', 0):.2f}")

        st.write("")
        with st.container(border=True):
            section("wave", "Pose preview", "eight sampled frames and the motion-energy curve")
            preview = state("preview")
            if preview and Path(preview).exists():
                st.image(preview, width="stretch")
            else:
                st.caption("Preview image not available for this run.")

        st.download_button(
            "Download motion data (NPZ)",
            (Path(motion_dir) / "motion.npz").read_bytes(),
            file_name="motion.npz",
            mime="application/octet-stream",
        )

        st.write("")
        with st.container(border=True):
            section("play", "Blender character render", "rigged mannequin, soundtrack muxed into the MP4")
            st.caption("Rendering takes several minutes; the rest of the interface stays usable afterwards.")
            if st.button("Render with Blender", type="primary"):
                path = render_video(Path(motion_dir), audio_path, mode)
                if path:
                    st.session_state["render_path"] = str(path)

            render_path = state("render_path")
            if render_path and Path(render_path).exists():
                st.video(render_path)
                st.download_button(
                    "Download MP4",
                    Path(render_path).read_bytes(),
                    file_name=Path(render_path).name,
                    mime="video/mp4",
                )


# ---------------------------------------------------------------------------
# Tab 4 — evaluation
# ---------------------------------------------------------------------------

with tabs[4]:
    section("chart", "Beat-only vs multi-level", "same library, same pipeline, different mapping")

    st.caption(
        "Both conditions run on this song with identical retrieval, blending and interface settings. "
        "Synchronisation is expected to be similar; expressiveness should favour multi-level mapping."
    )

    if st.button("Run both conditions", type="primary"):
        rows = []
        for condition in ("beat_only", "multi_level"):
            with st.status(f"Running {condition.replace('_', ' ')}…", expanded=False) as status:
                metadata = pipeline.run_song(
                    audio_path,
                    mode=condition,
                    rules=rules,
                    run_name=f"compare_{paths.slug(audio_path.stem)}_{condition}",
                )
                status.update(label=f"{condition.replace('_', ' ')} finished", state="complete")
            rows.append({
                "condition": condition,
                **{key: metadata["metrics"][key] for key in (
                    "bas", "beat_f1", "energy_corr", "beat_response_corr",
                    "dynamic_range", "clip_variety",
                )},
            })
        st.session_state["comparison_table"] = pd.DataFrame(rows).set_index("condition").round(3)

    comparison = state("comparison_table")
    if comparison is None or comparison.empty:
        empty_state("⚖", "No comparison yet", "Run both conditions to compare synchronisation and expressiveness.")
    else:
        table_col, chart_col = st.columns([1.15, 1.0], gap="large")
        with table_col:
            with st.container(border=True):
                section("chart", "Metrics", "beat alignment and expressiveness")
                st.dataframe(comparison, width="stretch")
                delta = comparison.loc["multi_level"] - comparison.loc["beat_only"]
                st.markdown("**Multi-level minus beat-only**")
                st.dataframe(delta.round(3).to_frame().T, width="stretch")
        with chart_col:
            with st.container(border=True):
                section("chart", "Side-by-side", "higher is better for all four metrics")
                melted = comparison.T.reset_index().rename(columns={"index": "metric"})
                melted = melted.melt(id_vars="metric", var_name="condition", value_name="value")
                bars = (
                    alt.Chart(melted)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0)),
                        xOffset="condition:N",
                        y=alt.Y("value:Q", title=None),
                        color=alt.Color(
                            "condition:N",
                            scale=alt.Scale(range=["#CFC9EE", "#6C5CE7"]),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                    )
                )
                st.altair_chart(altair_theme(bars, 270), width="stretch")

    st.write("")
    with st.container(border=True):
        section("book", "Blind A/B rating", "saved to outputs/user_study/responses.csv")
        existing = sorted((PROJECT_DIR / "outputs").glob("runs/*/character_*.mp4"))
        options = [str(path.relative_to(PROJECT_DIR)) for path in existing] or ["(no rendered clip yet)"]

        with st.form("rating_form", clear_on_submit=True):
            rater = st.text_input("Rater ID (anonymous code)", value="")
            a_col, b_col = st.columns(2)
            clip_a = a_col.selectbox("Clip A", options)
            clip_b = b_col.selectbox("Clip B", options, index=min(1, len(options) - 1))
            sync_a = a_col.slider("Clip A · synchronisation", 1, 5, 3)
            expr_a = a_col.slider("Clip A · expressiveness", 1, 5, 3)
            sync_b = b_col.slider("Clip B · synchronisation", 1, 5, 3)
            expr_b = b_col.slider("Clip B · expressiveness", 1, 5, 3)
            comment = st.text_area("Comment (optional)")
            submitted = st.form_submit_button("Save rating", type="primary")

        if submitted and rater:
            row = pd.DataFrame([{
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "rater": rater,
                "song": audio_path.name,
                "clip_a": clip_a,
                "clip_b": clip_b,
                "sync_a": sync_a,
                "expressiveness_a": expr_a,
                "sync_b": sync_b,
                "expressiveness_b": expr_b,
                "comment": comment,
            }])
            target = STUDY_DIR / "responses.csv"
            row.to_csv(target, mode="a", header=not target.exists(), index=False)
            st.success(f"Saved to {target.relative_to(PROJECT_DIR)}")


# ---------------------------------------------------------------------------
# Tab 5 — documentation
# ---------------------------------------------------------------------------

with tabs[5]:
    section("book", "How the system maps music to motion", "the five rules, the pipeline and the artifacts")

    with st.container(border=True):
        st.markdown(
            """
| Musical feature | Animation parameter | Implementation |
|---|---|---|
| Beat | when a movement starts | phrase slots on the beat grid |
| Tempo | how fast | retrieval prefers matching internal BPM; bounded speed scale |
| RMS energy | how strong | gain on pose amplitude and energy-family retrieval |
| Onset strength | how sharp the transition | blend duration between clips |
| Structure | which motion family | section energy rank restricts the candidate pool |
            """
        )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**Analysis**")
            st.caption("Librosa features on a 30 FPS grid, beat table, novelty-based sections.")
            st.markdown("**Library**")
            st.caption("1 408 AIST++ sequences → 16 737 two-second clips with nine descriptors.")
    with c2:
        with st.container(border=True):
            st.markdown("**Mapping**")
            st.caption("Explicit editable rules → one clip decision per phrase slot, with a text explanation.")
            st.markdown("**Blending**")
            st.caption("Time warping, Slerp, DTW alignment, stage anchoring and a ground pass.")
    with c3:
        with st.container(border=True):
            st.markdown("**Rendering**")
            st.caption("Rigged 21-bone mannequin, EEVEE, MP4 with the soundtrack, optional .blend/FBX.")
            st.markdown("**Evaluation**")
            st.caption("BAS, beat F1, energy and beat-response correlation, dynamic range, clip variety.")

    st.write("")
    st.markdown(
        "<div class='note'>Every run writes <code>plan.csv</code>, <code>mapping_explanation.txt</code>, "
        "<code>motion.npz</code>, <code>metrics.json</code> and <code>run.json</code>, so any reported "
        "number can be traced back to the decisions that produced it.</div>",
        unsafe_allow_html=True,
    )

    for name in ("DEVELOPMENT_LOG.md", "EXPERIMENT_REPORT.md", "PROGRESS_REPORT.md"):
        path = PROJECT_DIR / "docs" / name
        if path.exists():
            st.download_button(f"Download {name}", path.read_bytes(), file_name=name)
