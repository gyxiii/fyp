"""MusicMotion Studio — interactive music-driven character animation (CAT404).

A Streamlit front end for the pipeline in ``src/``:

    audio -> features -> mapping plan -> motion -> Blender render -> evaluation

The interface is deliberately *rule-transparent*: every mapping weight is
visible and editable, because inspecting and adjusting the mapping is the
difference between this system and a black-box dance generator.

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

import numpy as np
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
ACCENT_2 = "#00B8D9"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #1B1F3B;
    --muted: #6B7280;
    --line: #E6E8F0;
    --card: #FFFFFF;
    --accent: #6C5CE7;
    --accent-2: #00B8D9;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 500px at 8% -8%, rgba(108, 92, 231, 0.14), transparent 60%),
        radial-gradient(900px 420px at 100% 0%, rgba(0, 184, 217, 0.10), transparent 55%),
        #F6F7FB;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1240px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4 { letter-spacing: -0.02em; color: var(--ink); }
h1 { font-weight: 800; }
h2 { font-weight: 700; margin-top: 0.6rem; }
h3 { font-weight: 700; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151A32 0%, #10142A 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #E9ECF8; }
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 10px !important;
    color: #F2F4FF !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: #B7BEDC !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div { color: #E9ECF8; }

.brand {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.2rem 0 1rem 0;
}
.brand-mark {
    width: 42px; height: 42px; border-radius: 13px;
    display: grid; place-items: center; font-size: 1.25rem;
    background: linear-gradient(135deg, #6C5CE7, #00B8D9);
    box-shadow: 0 8px 20px rgba(108, 92, 231, 0.35);
}
.brand-name { font-weight: 800; font-size: 1.05rem; letter-spacing: -0.02em; line-height: 1.1; }
.brand-sub { font-size: 0.72rem; color: #A8B0D3; letter-spacing: 0.06em; text-transform: uppercase; }

/* ---------- hero ---------- */
.hero {
    position: relative;
    border-radius: 22px;
    padding: 1.7rem 1.9rem 1.55rem 1.9rem;
    background:
        radial-gradient(700px 240px at 100% 0%, rgba(0, 184, 217, 0.35), transparent 60%),
        linear-gradient(120deg, #1B1F3B 0%, #2A2F66 55%, #4B3FA8 100%);
    color: #FFFFFF;
    box-shadow: 0 24px 60px rgba(27, 31, 59, 0.28);
    overflow: hidden;
}
.hero:after {
    content: ""; position: absolute; right: -60px; bottom: -90px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.18), transparent 62%);
}
.kicker {
    font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #C9C3FF; font-weight: 700; margin-bottom: 0.45rem;
}
.hero h1 {
    color: #FFFFFF; font-size: 2.15rem; margin: 0 0 0.35rem 0; line-height: 1.12;
}
.hero p { color: #D7DAF2; font-size: 0.98rem; margin: 0; max-width: 780px; }
.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.95rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.3rem 0.7rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
    color: #F3F4FF;
}

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(27, 31, 59, 0.05);
    padding: 1.1rem 1.15rem;
}
.section-title {
    display: flex; align-items: baseline; gap: 0.55rem;
    margin: 0.2rem 0 0.15rem 0;
}
.section-title .t { font-size: 1.12rem; font-weight: 700; color: var(--ink); }
.section-title .h { font-size: 0.82rem; color: var(--muted); }
.section-rule { height: 2px; width: 46px; border-radius: 2px; margin: 0.35rem 0 0.9rem 0;
    background: linear-gradient(90deg, #6C5CE7, #00B8D9); }

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: 0 6px 18px rgba(27, 31, 59, 0.04);
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.03em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--ink); font-weight: 750; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem; background: rgba(255,255,255,0.75);
    padding: 0.35rem; border-radius: 14px; border: 1px solid var(--line);
    box-shadow: 0 6px 18px rgba(27, 31, 59, 0.04);
}
.stTabs [data-baseweb="tab"] {
    height: 40px; border-radius: 10px; padding: 0 1rem;
    font-weight: 600; color: #5A6079; background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C5CE7, #7C6BF0);
    color: #FFFFFF !important;
    box-shadow: 0 6px 14px rgba(108, 92, 231, 0.30);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- controls ---------- */
.stButton > button, [data-testid="stDownloadButton"] > button {
    border-radius: 11px; font-weight: 650; padding: 0.5rem 1.05rem;
    border: 1px solid var(--line); background: #FFFFFF; color: var(--ink);
    transition: all 0.15s ease;
}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    border-color: #C9C3FF; color: #4B3FA8; transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(108, 92, 231, 0.15);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7, #8B7CF6);
    color: #FFFFFF; border: none;
    box-shadow: 0 10px 22px rgba(108, 92, 231, 0.30);
}
.stButton > button[kind="primary"]:hover { box-shadow: 0 12px 26px rgba(108, 92, 231, 0.40); }

[data-testid="stExpander"] details {
    border: 1px solid var(--line); border-radius: 14px; background: #FFFFFF;
    box-shadow: 0 6px 16px rgba(27, 31, 59, 0.04);
}
[data-testid="stExpander"] summary { font-weight: 600; }

[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stDataFrame"] * { font-family: 'Inter', system-ui, sans-serif; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, #6C5CE7, #00B8D9); }

.empty {
    border: 1px dashed #C9CBE0; border-radius: 16px; padding: 1.6rem;
    text-align: center; color: var(--muted); background: rgba(255,255,255,0.6);
}
.empty .big { font-size: 1.7rem; margin-bottom: 0.35rem; }

.note {
    font-size: 0.82rem; color: var(--muted); line-height: 1.45;
}
.pill {
    display: inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.02em;
}
.pill-ok { background: #E7F8EF; color: #14804A; }
.pill-warn { background: #FFF4E5; color: #B25E09; }
.pill-info { background: #EEF0FF; color: #4B3FA8; }

footer { visibility: hidden; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #6C5CE7, #8B7CF6) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 9px !important;
}
section[data-testid="stSidebar"] .stCheckbox label, section[data-testid="stSidebar"] .stCheckbox p {
    color: #E9ECF8 !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #C9CBE0 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] { border-color: #8B7CF6 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: #6B7280; }
[data-testid="stMetricDelta"] svg { display: inline; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


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


# ---------------------------------------------------------------------------
# Small UI components
# ---------------------------------------------------------------------------

def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="kicker">CAT404 · Software Engineering Major Project</div>
            <h1>MusicMotion Studio</h1>
            <p>An interactive music-driven 3D character animation system built on multi-level musical
            features — beat, tempo, energy, onset and structure mapped onto motion, rendered in Blender,
            and measured with Beat Alignment Score.</p>
            <div class="chips">
                <span class="chip">◉ Multi-level mapping</span>
                <span class="chip">◫ 16 737 motion clips</span>
                <span class="chip">✧ Blender rendering</span>
                <span class="chip">◎ BAS evaluation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(icon: str, title: str, hint: str = "") -> None:
    hint_html = f"<span class='h'>{hint}</span>" if hint else ""
    st.markdown(
        f"<div class='section-title'><span class='t'>{icon} {title}</span>{hint_html}</div>"
        "<div class='section-rule'></div>",
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, text: str) -> None:
    st.markdown(
        f"<div class='empty'><div class='big'>{icon}</div>"
        f"<div style='font-weight:700;color:#1B1F3B;margin-bottom:.25rem'>{title}</div>"
        f"<div class='note'>{text}</div></div>",
        unsafe_allow_html=True,
    )


def rules_from_sidebar(base: dict, mode: str) -> dict:
    rules = json.loads(json.dumps(base))
    rules["mode"] = mode

    st.sidebar.markdown("#### Mapping weights")
    st.sidebar.caption("Every value below is part of the mapping that will be recorded in the run.")

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

    st.markdown("#### Music input")
    uploaded = st.file_uploader("Upload a WAV file", type=["wav"], label_visibility="collapsed")
    demos = sorted(path.name for path in paths.audio_files())
    demo_choice = st.selectbox("or use a bundled song", demos, index=0 if demos else None)

    st.divider()
    st.markdown("#### Mapping mode")
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
    st.caption(
        "All conditions share the same motion library, retrieval, blending and interface — "
        "only the active mapping layers differ."
    )

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
song_col, audio_col = st.columns([2.1, 1.0], gap="large")
with song_col:
    st.markdown(
        f"""
        <div class="note" style="text-transform:uppercase;letter-spacing:.12em;font-weight:700;color:#6B7280">
            Now working on
        </div>
        <div style="font-size:1.35rem;font-weight:800;color:#1B1F3B;line-height:1.2">{audio_path.name}</div>
        <div class="note" style="margin-top:.2rem">Mode: <b>{mode}</b> · mapping rules editable in the sidebar</div>
        """,
        unsafe_allow_html=True,
    )
with audio_col:
    st.audio(str(audio_path))

tabs = st.tabs([
    "① Music analysis",
    "② Mapping plan",
    "③ Motion & render",
    "④ Evaluation",
    "⑤ Documentation",
])


# ---------------------------------------------------------------------------
# Tab 1 — analysis
# ---------------------------------------------------------------------------

with tabs[0]:
    section("🎼", "Musical analysis", "beat · tempo · energy · onset · structure")

    if st.button("Re-analyse music", type="secondary"):
        reset_results()
        analyse(audio_path, refresh=True)

    analysis = state("analysis")
    if analysis is None or state("analysis_key", (None,))[1] != str(audio_path):
        analysis = analyse(audio_path)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Duration", f"{analysis.duration:.1f} s")
    c2.metric("Tempo", f"{analysis.tempo_bpm:.1f} BPM")
    c3.metric("Detected beats", len(analysis.beats))
    c4.metric("Sections", len(analysis.sections))

    st.write("")
    with st.container(border=True):
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
            section("▤", "Sections", "energy rank drives the motion family")
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
            section("◷", "Beat features", "the table the mapping actually consumes")
            st.dataframe(
                analysis.beats[[
                    "beat_time", "bpm_local", "energy_norm", "onset_norm", "section_label"
                ]].round(3),
                width="stretch",
                hide_index=True,
                height=300,
            )


# ---------------------------------------------------------------------------
# Tab 2 — plan
# ---------------------------------------------------------------------------

with tabs[1]:
    section("🗺", "Music-to-motion mapping", "one clip decision per musical phrase")

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
            section("▦", "Slot decisions", f"{len(plan.plan)} slots · mode {mode}")
            st.dataframe(
                plan.plan[[
                    "slot", "start_time", "end_time", "bpm", "rate", "gain",
                    "blend_seconds", "energy_target", "section_label", "clip_id",
                    "clip_genre", "clip_tempo_bpm",
                ]].round(3),
                width="stretch",
                hide_index=True,
                height=360,
            )

        chart_col, info_col = st.columns([1.6, 1.0], gap="large")
        with chart_col:
            with st.container(border=True):
                section("📈", "Intensity over time", "gain, playback rate and blend length")
                st.line_chart(
                    plan.plan.set_index("start_time")[["gain", "rate"]],
                    height=260,
                    color=["#6C5CE7", "#00B8D9"],
                )
        with info_col:
            with st.container(border=True):
                section("◎", "Plan summary", "stored with the run")
                st.write(plan.summary.get("genre_distribution", {}))
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

with tabs[2]:
    section("🎬", "Motion generation", "Slerp blending · DTW alignment · ground correction")

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
            "<div class='note'>The motion is assembled in direction space: clips are time-warped, "
            "scaled by the energy gain and blended with spherical interpolation. A ground pass keeps "
            "the feet on the floor.</div>",
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
            section("🖼", "Pose preview", "eight sampled frames and the motion-energy curve")
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
            section("🎞", "Blender character render", "rigged mannequin, soundtrack muxed into the MP4")
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

with tabs[3]:
    section("📊", "Beat-only vs multi-level", "same library, same pipeline, different mapping")

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
                section("▤", "Metrics", "beat alignment and expressiveness")
                st.dataframe(comparison, width="stretch")
                delta = comparison.loc["multi_level"] - comparison.loc["beat_only"]
                st.markdown("**Multi-level minus beat-only**")
                st.dataframe(delta.round(3).to_frame().T, width="stretch")
        with chart_col:
            with st.container(border=True):
                section("📈", "Side-by-side", "higher is better for all four metrics")
                chart_df = comparison.T.reset_index().rename(columns={"index": "metric"})
                st.bar_chart(
                    chart_df.set_index("metric"),
                    height=280,
                    color=["#9AA0B5", "#6C5CE7"],
                )

    st.write("")
    with st.container(border=True):
        section("🎚", "Blind A/B rating", "user study · saved to outputs/user_study/responses.csv")
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

with tabs[4]:
    section("📚", "How the system maps music to motion", "the five rules, the pipeline and the artifacts")

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
