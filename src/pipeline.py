"""End-to-end pipeline for one song and one mapping condition.

    audio -> features -> plan -> motion -> metrics -> (optional) Blender render

This is the single entry point used by the experiment runner, the interactive
application and the command line.  Every stage writes its artefacts into one run
folder so a result can always be traced back to its inputs:

    outputs/runs/<run>/
        plan.csv               per-slot mapping decisions
        mapping_beats.csv      beat-level musical features used by the mapping
        mapping_explanation.txt one human-readable line per decision
        motion.npz             assembled motion (root, directions, nodes, joints)
        motion_preview.png     static quality check
        metrics.json           BAS / expressiveness metrics
        run.json               provenance (mode, rules, library, timings)
        character.mp4          optional Blender render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

try:
    from . import beat_alignment, mapping, motion_blend, music_features, paths
except ImportError:  # pragma: no cover - script execution
    import beat_alignment
    import mapping
    import motion_blend
    import music_features
    import paths


def ensure_features(audio_path: Path, refresh: bool = False) -> "music_features.MusicAnalysis":
    """Analyse the song once and cache the feature tables on disk."""
    target = paths.feature_dir(audio_path)
    summary = target / "summary.json"
    if not refresh and summary.exists():
        return music_features.MusicAnalysis.load(target, audio_path)
    return music_features.analyse_music(audio_path, out_dir=target)


def run_song(
    audio_path: Path,
    mode: str = "multi_level",
    rules: dict | None = None,
    run_name: str | None = None,
    fps: int = 30,
    render: bool = False,
    refresh_features: bool = False,
    rules_path: Path | None = None,
) -> dict:
    """Execute the full pipeline for one song and one mapping mode."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    rules = rules or mapping.load_rules(rules_path)
    run_name = run_name or f"{paths.slug(audio_path.stem)}_{mode}"
    run_dir = paths.run_dir(run_name)

    started = time.time()
    timings: dict[str, float] = {}

    timings_start = time.time()
    analysis = ensure_features(audio_path, refresh=refresh_features)
    timings["analysis"] = time.time() - timings_start

    if not paths.LIBRARY_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Motion library not found. Run `python src/motion_library.py` first."
        )
    library_index = pd.read_csv(paths.LIBRARY_INDEX_PATH)

    timings_start = time.time()
    plan = mapping.build_plan(
        beats=analysis.beats,
        library=library_index,
        rules=rules,
        mode=mode,
        duration=analysis.duration,
    )
    plan.save(run_dir)
    timings["mapping"] = time.time() - timings_start

    timings_start = time.time()
    library = motion_blend.ClipLibrary()
    motion = motion_blend.build_motion(plan.plan, library, rules, fps=fps)
    motion.save(run_dir)
    timings["motion"] = time.time() - timings_start

    timings_start = time.time()
    metrics = beat_alignment.evaluate_run(
        run_dir, feature_dir=paths.feature_dir(audio_path)
    )
    timings["metrics"] = time.time() - timings_start

    render_info = None
    if render:
        render_info = render_run(run_dir, audio_path, fps=fps)
        timings["render"] = render_info.get("seconds", 0.0)

    metadata = {
        "audio": str(audio_path),
        "mode": mode,
        "run": run_name,
        "run_dir": str(run_dir),
        "fps": fps,
        "duration": analysis.duration,
        "tempo_bpm": analysis.tempo_bpm,
        "beats": int(len(analysis.beats)),
        "sections": len(analysis.sections),
        "slots": int(len(plan.plan)),
        "metrics": metrics,
        "render": render_info,
        "timings": timings,
        "total_seconds": time.time() - started,
        "features": str(paths.feature_dir(audio_path)),
        "library": str(paths.LIBRARY_DIR),
        "rules": plan.rules,
    }
    paths.save_json(run_dir / "run.json", metadata)

    return metadata


def render_run(run_dir: Path, audio_path: Path, fps: int = 30) -> dict:
    """Render the assembled motion with Blender (headless)."""
    import subprocess

    run_dir = Path(run_dir)
    motion_path = run_dir / "motion.npz"
    output_path = run_dir / "character.mp4"
    script = paths.BLENDER_DIR / "render_character.py"

    if not paths.BLENDER_EXE.is_file():
        raise FileNotFoundError(f"Blender not found: {paths.BLENDER_EXE}")

    command = [
        str(paths.BLENDER_EXE),
        "--background",
        "--factory-startup",
        "--python",
        str(script),
        "--",
        "--motion", str(motion_path),
        "--audio", str(audio_path),
        "--output", str(output_path),
        "--blend-out", str(run_dir / "character.blend"),
    ]

    started = time.time()
    result = subprocess.run(
        command,
        cwd=str(paths.PROJECT_DIR),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=3600,
    )
    (run_dir / "render.log").write_text(
        (result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8"
    )
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "Blender render failed. See outputs render.log for details."
        )

    return {
        "output": str(output_path),
        "seconds": time.time() - started,
        "size_mb": output_path.stat().st_size / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full music-to-animation pipeline.")
    parser.add_argument("audio", help="audio file to animate")
    parser.add_argument("--mode", default="multi_level", choices=mapping.MODES)
    parser.add_argument("--rules", default=str(paths.RULES_PATH))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--render", action="store_true", help="render with Blender")
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args()

    metadata = run_song(
        Path(args.audio),
        mode=args.mode,
        run_name=args.run_name,
        render=args.render,
        refresh_features=args.refresh_features,
        rules_path=Path(args.rules),
    )

    print(f"Run      : {metadata['run']} ({metadata['run_dir']})")
    print(f"Mode     : {metadata['mode']}")
    print(f"Slots    : {metadata['slots']}")
    print(f"Metrics  : BAS={metadata['metrics']['bas']:.3f} "
          f"F1={metadata['metrics']['beat_f1']:.3f} "
          f"energy_corr={metadata['metrics']['energy_corr']:.3f}")
    print(f"Total    : {metadata['total_seconds']:.1f} s")
    if metadata.get("render"):
        print(f"Video    : {metadata['render']['output']}")


if __name__ == "__main__":
    main()
