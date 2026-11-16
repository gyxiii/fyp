"""Central path resolution for the CAT404 music-driven animation project.

The large AIST++ dataset is *not* part of the source tree.  By default the
pipeline reads it from ``<project>/data``.  A different location can be used
without moving files:

    set CAT404_DATA_ROOT=C:\\Users\\me\\Desktop\\CAT404_music_animation_starter\\data

Only ``data`` lives outside the repository; all code, configuration, reports
and small results stay inside the project.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

_data_override = os.environ.get("CAT404_DATA_ROOT")
DATA_DIR = (
    Path(_data_override).expanduser().resolve()
    if _data_override
    else PROJECT_DIR / "data"
)

AUDIO_DIR = DATA_DIR / "audio"
FEATURE_DIR = DATA_DIR / "features"
LIBRARY_DIR = DATA_DIR / "library"

_aist_override = os.environ.get("CAT404_AIST_ROOT")
AIST_DIR = (
    Path(_aist_override).expanduser().resolve()
    if _aist_override
    else DATA_DIR / "aistplusplus" / "keypoints3d"
)

CONFIG_DIR = PROJECT_DIR / "config"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUT_DIR = PROJECT_DIR / "outputs"
BLENDER_DIR = PROJECT_DIR / "blender"
RULES_PATH = CONFIG_DIR / "mapping_rules.json"
LIBRARY_PATH = LIBRARY_DIR / "motion_library.npz"
LIBRARY_INDEX_PATH = LIBRARY_DIR / "motion_index.csv"
LIBRARY_SUMMARY_PATH = LIBRARY_DIR / "library_summary.json"

BLENDER_EXE = Path(
    os.environ.get(
        "CAT404_BLENDER",
        r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    )
)

FPS = 30
SAMPLE_RATE = 22050


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for folder in (AUDIO_DIR, FEATURE_DIR, LIBRARY_DIR, OUTPUT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    """Turn a filename or title into a filesystem-safe identifier."""
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("._")
    return cleaned or "untitled"


def feature_dir(audio_path: Path) -> Path:
    """Per-song feature folder: data/features/<audio stem>/."""
    return FEATURE_DIR / slug(Path(audio_path).stem)


def run_dir(name: str) -> Path:
    """Folder for one generation run: outputs/runs/<name>/."""
    path = OUTPUT_DIR / "runs" / slug(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def audio_files() -> list[Path]:
    """Audio files available to batch experiments (wav/flac/mp3/ogg)."""
    if not AUDIO_DIR.exists():
        return []
    patterns = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")
    found: list[Path] = []
    for pattern in patterns:
        found.extend(AUDIO_DIR.glob(pattern))
    return sorted(found)
