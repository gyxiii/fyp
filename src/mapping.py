"""Multi-level music-to-motion mapping (proposal section 5.3).

The mapping is deliberately layered, explicit and inspectable:

    beat       -> slot boundaries (when a movement starts)
    tempo      -> which motion tempo is retrieved, plus its playback rate
    energy     -> motion gain (how strong) and the energy family retrieved
    onset      -> transition length (sharp attack -> short blend)
    structure  -> the motion family (calm / mid / energetic section)

Each slot is a small musical phrase (``slot_beats`` beats).  One motion clip is
retrieved per slot from the library index built by ``motion_library.py``.

Four rule sets are supported for the controlled experiment:

    multi_level    all features active
    beat_only      only beat timing; tempo/energy/onset/structure neutral
    no_tempo       ablation: tempo replaced by the reference tempo
    no_energy      ablation: energy replaced by the neutral value
    no_onset       ablation: transition length fixed
    no_structure   ablation: no motion-family constraint

All variants share the same library, retrieval code, blending code and
interface, which is what makes the comparison in the proposal fair.

Run:

    python src/mapping.py --features data/features/test_song
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import paths
except ImportError:  # pragma: no cover - script execution
    import paths


MODES = ("multi_level", "beat_only", "no_tempo", "no_energy", "no_onset", "no_structure")

_FEATURE_FLAGS = {
    "multi_level": {"tempo": True, "energy": True, "onset": True, "structure": True},
    "beat_only": {"tempo": False, "energy": False, "onset": False, "structure": False},
    "no_tempo": {"tempo": False, "energy": True, "onset": True, "structure": True},
    "no_energy": {"tempo": True, "energy": False, "onset": True, "structure": True},
    "no_onset": {"tempo": True, "energy": True, "onset": False, "structure": True},
    "no_structure": {"tempo": True, "energy": True, "onset": True, "structure": False},
}

NEUTRAL_ENERGY = 0.5
NEUTRAL_TEMPO_NORM = 0.5


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def load_rules(path: Path | None = None) -> dict:
    rules = paths.load_json(Path(path or paths.RULES_PATH), None)
    if rules is None:
        raise FileNotFoundError(f"Mapping rules not found: {path or paths.RULES_PATH}")
    return rules


def effective_rules(base: dict, mode: str) -> dict:
    """Return a copy of ``base`` with the features of ``mode`` enabled/disabled."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    rules = copy.deepcopy(base)
    rules["mode"] = mode

    for feature, flag in _FEATURE_FLAGS[mode].items():
        rules.setdefault(feature, {})["enabled"] = bool(flag)

    # Beat-only keeps the beat grid but neutralises every expressive decision.
    if mode == "beat_only":
        rules["selection"]["diversity_penalty"] = rules["selection"].get("diversity_penalty", 0.15)

    return rules


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * np.clip(t, 0.0, 1.0))


def _jitter(slot: int, clip_id: int, scale: float) -> float:
    """Deterministic pseudo-random tie breaker (reproducible experiments)."""
    if scale <= 0:
        return 0.0
    digest = hashlib.md5(f"{slot}:{clip_id}".encode()).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return (value - 0.5) * scale


# ---------------------------------------------------------------------------
# Plan container
# ---------------------------------------------------------------------------

@dataclass
class MotionPlan:
    mode: str
    rules: dict
    plan: pd.DataFrame
    beats: pd.DataFrame
    summary: dict = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)

    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.plan.to_csv(out_dir / "plan.csv", index=False)
        self.beats.to_csv(out_dir / "mapping_beats.csv", index=False)
        paths.save_json(out_dir / "mapping_summary.json", self.summary)
        paths.save_json(out_dir / "mapping_rules_used.json", self.rules)
        with (out_dir / "mapping_explanation.txt").open("w", encoding="utf-8") as handle:
            handle.write("\n".join(self.explanation))
        return out_dir

    def to_events(self) -> pd.DataFrame:
        """Backwards-compatible beat/event table for the simple renderers."""
        return pd.DataFrame({
            "time_sec": self.plan["start_time"],
            "event_type": "motion",
            "intensity": self.plan["gain"],
        })


# ---------------------------------------------------------------------------
# Slotting
# ---------------------------------------------------------------------------

def build_slots(beats: pd.DataFrame, slot_beats: int) -> list[tuple[int, int]]:
    """Split beats into phrase slots, aligned to the first downbeat when known."""
    count = len(beats)
    if count == 0:
        return []

    phase = 0
    if "is_downbeat" in beats.columns:
        downbeats = np.flatnonzero(beats["is_downbeat"].to_numpy() == 1)
        if downbeats.size:
            phase = int(downbeats[0]) % max(slot_beats, 1)

    boundaries = [0]
    start = phase if phase > 0 else slot_beats
    for index in range(start, count, max(slot_beats, 1)):
        if index > boundaries[-1]:
            boundaries.append(index)
    if boundaries[-1] != count:
        boundaries.append(count)

    slots = []
    for start_index, end_index in zip(boundaries[:-1], boundaries[1:]):
        if end_index > start_index:
            slots.append((start_index, end_index))
    return slots


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve(
    library: pd.DataFrame,
    target_energy: float,
    target_tempo_norm: float,
    target_class: int | None,
    rules: dict,
    slot_index: int,
    recent_clips: list[int],
    recent_sources: list[str],
) -> pd.Series:
    """Rule-based nearest-neighbour retrieval with a diversity penalty."""
    selection = rules.get("selection", {})
    tempo_rules = rules.get("tempo", {})
    energy_rules = rules.get("energy", {})

    pool = library
    if target_class is not None and rules.get("structure", {}).get("enabled", True):
        class_pool = library[library["energy_class"] == int(target_class)]
        if len(class_pool) >= int(selection.get("min_pool", 12)):
            pool = class_pool

    weights = {
        "tempo": float(tempo_rules.get("weight", 0.55)) if tempo_rules.get("enabled", True) else 0.0,
        "energy": float(energy_rules.get("weight", 0.75)) if energy_rules.get("enabled", True) else 0.0,
        "rhythm": float(selection.get("rhythmicity_weight", 0.2)),
    }

    scores = (
        weights["tempo"] * (pool["tempo_n"] - target_tempo_norm).abs()
        + weights["energy"] * (pool["energy_n"] - target_energy).abs()
        + weights["rhythm"] * (1.0 - pool["rhythmicity_n"])
    )

    memory = int(selection.get("memory_slots", 6))
    recent = set(recent_clips[-memory:]) if memory > 0 else set()
    if recent:
        scores = scores + pool["clip_id"].isin(recent).astype(float) * float(
            selection.get("diversity_penalty", 0.15)
        )

    genre_memory = int(selection.get("genre_memory", 0) or 0)
    if genre_memory > 0 and recent_sources:
        sources = set(recent_sources[-genre_memory:])
        scores = scores + pool["source_file"].isin(sources).astype(float) * float(
            selection.get("genre_penalty", 0.10)
        )

    jitter = float(selection.get("jitter", 0.01))
    if jitter > 0:
        scores = scores + [
            _jitter(slot_index, int(clip_id), jitter)
            for clip_id in pool["clip_id"].to_numpy()
        ]

    best_index = int(np.argmin(scores.to_numpy()))
    return pool.iloc[best_index]


# ---------------------------------------------------------------------------
# Main mapping
# ---------------------------------------------------------------------------

def build_plan(
    beats: pd.DataFrame,
    library: pd.DataFrame,
    rules: dict,
    mode: str = "multi_level",
    duration: float | None = None,
) -> MotionPlan:
    """Turn beat features into a sequence of motion-clip decisions."""
    active = effective_rules(rules, mode)
    tempo_rules = active.get("tempo", {})
    energy_rules = active.get("energy", {})
    onset_rules = active.get("onset", {})
    structure_rules = active.get("structure", {})

    slot_beats = int(active.get("slot_beats", 2))
    slots = build_slots(beats, slot_beats)
    if not slots:
        raise ValueError("No beats available for mapping.")

    if duration is None:
        duration = float(beats["beat_time"].iloc[-1] + 1.0)

    ref_bpm = float(tempo_rules.get("ref_bpm", 120.0))
    min_bpm = float(tempo_rules.get("min_bpm", 60.0))
    max_bpm = float(tempo_rules.get("max_bpm", 180.0))

    rows: list[dict] = []
    explanation: list[str] = []
    recent_clips: list[int] = []
    recent_sources: list[str] = []

    for slot_index, (start_beat, end_beat) in enumerate(slots):
        slot_beats_df = beats.iloc[start_beat:end_beat]
        start_time = float(slot_beats_df["beat_time"].iloc[0])
        if end_beat < len(beats):
            end_time = float(beats["beat_time"].iloc[end_beat])
        else:
            end_time = float(min(duration, start_time + 2.0 * float(slot_beats_df["ioi"].median())))

        if end_time - start_time < 0.20:
            continue

        bpm = float(np.median(slot_beats_df["bpm_local"])) if "bpm_local" in slot_beats_df else ref_bpm
        energy = float(np.clip(np.mean(slot_beats_df.get("energy_norm", NEUTRAL_ENERGY)), 0.0, 1.0))
        onset = float(np.clip(np.mean(slot_beats_df.get("onset_norm", 0.5)), 0.0, 1.0))
        section_class = int(round(float(np.mean(slot_beats_df.get("section_class", 1)))))
        section_label = str(slot_beats_df.get("section_label", pd.Series(["A"])).mode().iloc[0])

        if tempo_rules.get("enabled", True):
            rate = float(np.clip(
                (bpm / max(ref_bpm, 1.0)) ** float(tempo_rules.get("exponent", 1.0)),
                float(tempo_rules.get("min_rate", 0.75)),
                float(tempo_rules.get("max_rate", 1.40)),
            ))
            tempo_target_norm = float(np.clip((bpm - min_bpm) / max(max_bpm - min_bpm, 1e-6), 0.0, 1.0))
        else:
            bpm = ref_bpm
            rate = 1.0
            tempo_target_norm = NEUTRAL_TEMPO_NORM

        if energy_rules.get("enabled", True):
            gain = lerp(
                float(energy_rules.get("min_gain", 0.55)),
                float(energy_rules.get("max_gain", 1.45)),
                energy,
            )
            energy_target = energy
        else:
            gain = 1.0
            energy_target = NEUTRAL_ENERGY

        if onset_rules.get("enabled", True):
            blend_seconds = lerp(
                float(onset_rules.get("max_blend", 0.40)),
                float(onset_rules.get("min_blend", 0.08)),
                onset,
            )
        else:
            blend_seconds = float(onset_rules.get("default_blend", 0.30))

        blend_seconds = float(min(
            max(blend_seconds, 0.15),
            max((end_time - start_time) * 0.45, 0.02),
            float(active.get("blend", {}).get("max_blend_seconds", 0.5)),
        ))

        class_target = section_class if structure_rules.get("enabled", True) else None

        clip = _retrieve(
            library=library,
            target_energy=energy_target,
            target_tempo_norm=tempo_target_norm,
            target_class=class_target,
            rules=active,
            slot_index=slot_index,
            recent_clips=recent_clips,
            recent_sources=recent_sources,
        )

        recent_clips.append(int(clip["clip_id"]))
        recent_sources.append(str(clip["source_file"]))

        rows.append({
            "slot": slot_index,
            "start_beat": int(start_beat),
            "end_beat": int(end_beat),
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "bpm": bpm,
            "rate": rate,
            "gain": gain,
            "blend_seconds": blend_seconds,
            "energy_target": energy_target,
            "onset_target": onset,
            "section_class": section_class,
            "section_label": section_label,
            "clip_id": int(clip["clip_id"]),
            "clip_genre": str(clip["genre"]),
            "clip_source": str(clip["source_file"]),
            "clip_energy": float(clip["energy"]),
            "clip_energy_class": int(clip["energy_class"]),
            "clip_tempo_bpm": float(clip["internal_bpm"]),
            "clip_rhythmicity": float(clip["rhythmicity"]),
            "clip_score": float(
                float(active.get("tempo", {}).get("weight", 0.55)) * abs(float(clip["tempo_n"]) - tempo_target_norm)
                + float(active.get("energy", {}).get("weight", 0.75)) * abs(float(clip["energy_n"]) - energy_target)
            ),
        })

        explanation.append(
            f"slot {slot_index:02d} | {start_time:6.2f}-{end_time:6.2f}s | "
            f"{len(slot_beats_df)} beats @ {bpm:5.1f} BPM | "
            f"energy={energy:.2f} onset={onset:.2f} section={section_label}(class {section_class}) | "
            f"rate={rate:.2f} gain={gain:.2f} blend={blend_seconds:.2f}s | "
            f"clip {int(clip['clip_id'])} ({clip['genre']}, {clip['source_file']}, "
            f"e={float(clip['energy']):.4f}, tempo={float(clip['internal_bpm']):.0f} BPM)"
        )

    plan = pd.DataFrame(rows)
    if plan.empty:
        raise ValueError("The generated plan is empty.")

    summary = {
        "mode": mode,
        "song_beats": int(len(beats)),
        "slots": int(len(plan)),
        "slot_beats": slot_beats,
        "duration": float(duration),
        "mean_rate": float(plan["rate"].mean()),
        "mean_gain": float(plan["gain"].mean()),
        "mean_blend": float(plan["blend_seconds"].mean()),
        "unique_clips": int(plan["clip_id"].nunique()),
        "unique_sources": int(plan["clip_source"].nunique()),
        "genre_distribution": {
            str(key): int(value)
            for key, value in plan["clip_genre"].value_counts().items()
        },
        "energy_class_distribution": {
            str(key): int(value)
            for key, value in plan["clip_energy_class"].value_counts().sort_index().items()
        },
        "features_active": _FEATURE_FLAGS[mode],
        "rules": active,
    }

    return MotionPlan(
        mode=mode,
        rules=active,
        plan=plan,
        beats=beats.copy(),
        summary=summary,
        explanation=explanation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Map musical features to motion-clip decisions.")
    parser.add_argument("--features", required=True, help="Feature folder produced by music_features.py")
    parser.add_argument("--library", default=str(paths.LIBRARY_INDEX_PATH))
    parser.add_argument("--rules", default=str(paths.RULES_PATH))
    parser.add_argument("--mode", default="multi_level", choices=MODES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from music_features import MusicAnalysis

    analysis = MusicAnalysis.load(Path(args.features))
    library = pd.read_csv(args.library)
    rules = load_rules(Path(args.rules))

    plan = build_plan(
        beats=analysis.beats,
        library=library,
        rules=rules,
        mode=args.mode,
        duration=analysis.duration,
    )

    out_dir = Path(args.out) if args.out else Path(args.features) / f"plan_{args.mode}"
    plan.save(out_dir)

    print(f"Mode        : {plan.mode}")
    print(f"Slots       : {len(plan.plan)}")
    print(f"Mean rate   : {plan.summary['mean_rate']:.3f}")
    print(f"Mean gain   : {plan.summary['mean_gain']:.3f}")
    print(f"Mean blend  : {plan.summary['mean_blend']:.3f}s")
    print(f"Unique clips: {plan.summary['unique_clips']}")
    print(f"Output      : {out_dir}")
    print("\nFirst 5 decisions:")
    for line in plan.explanation[:5]:
        print("  " + line)


if __name__ == "__main__":
    main()
