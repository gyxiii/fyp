"""Beat alignment and expressiveness metrics (proposal section 8 / RQ1-RQ3).

The evaluation compares the generated motion against the music it was driven
by.  Two families of metrics are used:

Synchronisation
    BAS-Gaussian   mean over music beats of exp(-d^2 / (2 sigma^2)), where d is
                   the distance to the closest kinematic beat of the motion
                   (sigma = 0.15 s, about a 16th note at 100 BPM)
    beat F1        harmonic mean of motion-beat precision and music-beat recall
                   within a 0.15 s tolerance
    response corr  correlation between per-beat motion energy and per-beat
                   musical energy (does the character accent strong beats?)

Expressiveness
    energy corr    Pearson correlation between the motion-energy curve and the
                   music RMS curve (resampled to the animation frame rate)
    dynamic range  90th / 10th percentile of the motion-energy curve
    clip variety   unique clips / slots in the plan

Implementation notes
    Kinematic beats are detected from the mean joint-speed signal with a
    simple adaptive peak picker, which keeps the metric independent of the
    music analysis (no Librosa beat tracker is applied to the motion).

Run:

    python src/beat_alignment.py --run outputs/runs/smoke_multilevel --features data/features/test_song
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from . import paths
except ImportError:  # pragma: no cover - script execution
    import paths


DEFAULT_TOLERANCE = 0.15
DEFAULT_SIGMA = 0.15


# ---------------------------------------------------------------------------
# Beat extraction from motion
# ---------------------------------------------------------------------------

def motion_speed_signal(joints: np.ndarray, fps: int = 30) -> np.ndarray:
    """Mean joint speed in metres per second, one value per frame."""
    velocity = np.diff(joints, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=-1).mean(axis=1)
    return np.concatenate([[0.0], speed])


def kinematic_beats(
    speed: np.ndarray,
    fps: int = 30,
    min_gap: float = 0.20,
) -> np.ndarray:
    """Detect motion beats as local maxima of the speed signal."""
    if speed.size < 5:
        return np.array([])

    # Light smoothing avoids two peaks from a single acceleration spike.
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(speed, kernel, mode="same")

    threshold = float(np.mean(smoothed) + 0.5 * np.std(smoothed))
    min_distance = max(int(min_gap * fps), 1)

    peaks: list[int] = []
    last = -min_distance
    for index in range(1, len(smoothed) - 1):
        if smoothed[index] < threshold:
            continue
        if smoothed[index] < smoothed[index - 1] or smoothed[index] < smoothed[index + 1]:
            continue
        if index - last < min_distance:
            continue
        peaks.append(index)
        last = index

    return np.array(peaks, dtype=float) / fps


# ---------------------------------------------------------------------------
# Alignment metrics
# ---------------------------------------------------------------------------

def nearest_distances(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Distance from every ``query`` point to the closest ``reference`` point."""
    if reference.size == 0 or query.size == 0:
        return np.array([])
    indices = np.abs(query[:, None] - reference[None, :]).argmin(axis=1)
    return np.abs(query - reference[indices])


def beat_metrics(
    music_beats: np.ndarray,
    motion_beats: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    sigma: float = DEFAULT_SIGMA,
) -> dict:
    """BAS-Gaussian, precision, recall and F1 for one animation."""
    music_beats = np.asarray(music_beats, dtype=float)
    motion_beats = np.asarray(motion_beats, dtype=float)

    if music_beats.size == 0 or motion_beats.size == 0:
        return {
            "bas": 0.0,
            "beat_precision": 0.0,
            "beat_recall": 0.0,
            "beat_f1": 0.0,
            "motion_beats": int(motion_beats.size),
        }

    distances = nearest_distances(music_beats, motion_beats)
    bas = float(np.mean(np.exp(-(distances**2) / (2.0 * sigma**2))))
    precision = float(np.mean(distances <= tolerance))

    distances_back = nearest_distances(motion_beats, music_beats)
    recall = float(np.mean(distances_back <= tolerance))

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "bas": bas,
        "beat_precision": precision,
        "beat_recall": recall,
        "beat_f1": float(f1),
        "motion_beats": int(motion_beats.size),
        "mean_distance": float(np.mean(distances)),
    }


# ---------------------------------------------------------------------------
# Expressiveness metrics
# ---------------------------------------------------------------------------

def energy_correlation(
    motion_energy: np.ndarray,
    music_energy: np.ndarray,
    music_times: np.ndarray,
    fps: int = 30,
) -> float:
    """Correlation between the motion-energy curve and the music RMS curve."""
    if motion_energy.size < 10 or music_energy.size < 10:
        return 0.0

    motion_times = np.arange(motion_energy.size) / fps
    music_on_motion = np.interp(motion_times, music_times, music_energy)
    motion = motion_energy - motion_energy.mean()
    music = music_on_motion - music_on_motion.mean()

    denominator = np.linalg.norm(motion) * np.linalg.norm(music)
    if denominator < 1e-9:
        return 0.0
    return float(np.dot(motion, music) / denominator)


def beat_response(
    joints: np.ndarray,
    music_beats: np.ndarray,
    beat_energy: np.ndarray,
    fps: int = 30,
) -> tuple[float, float]:
    """Correlate per-beat motion energy with per-beat musical energy.

    For every music beat the mean joint speed in a short window after the beat
    is compared with the energy of that beat.  A high correlation means the
    character moves more strongly on musically strong beats.
    """
    speed = motion_speed_signal(joints, fps)
    if speed.size < 5 or music_beats.size < 4:
        return 0.0, 0.0

    window = max(int(0.12 * fps), 2)
    responses = []
    energies = []
    for time_sec, energy in zip(music_beats, beat_energy):
        start = int(time_sec * fps)
        end = min(start + window, speed.size)
        if start >= speed.size:
            continue
        responses.append(float(np.mean(speed[start:end])))
        energies.append(float(energy))

    if len(responses) < 4 or np.std(responses) < 1e-9 or np.std(energies) < 1e-9:
        return 0.0, 0.0

    correlation = float(np.corrcoef(responses, energies)[0, 1])
    return correlation, float(np.mean(responses))


def dynamic_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return 1.0
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    if low < 1e-6:
        return float(high / 1e-6)
    return float(high / low)


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    feature_dir: Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    """Compute every metric for one generated run and save ``metrics.json``."""
    from motion_blend import Motion
    from music_features import MusicAnalysis

    run_dir = Path(run_dir)
    motion = Motion.load(run_dir)
    fps = motion.fps

    analysis = MusicAnalysis.load(Path(feature_dir)) if feature_dir else None
    if analysis is None:
        raise ValueError("A feature folder is required to evaluate a run.")

    beats = analysis.beats
    music_beats = beats["beat_time"].to_numpy(dtype=float)
    beat_energy = beats["energy_norm"].to_numpy(dtype=float)

    speed = motion_speed_signal(motion.joints, fps)
    detected = kinematic_beats(speed, fps)

    metrics = beat_metrics(music_beats, detected, tolerance=tolerance)
    metrics["motion_energy_mean"] = float(motion.energy.mean())
    metrics["motion_energy_std"] = float(motion.energy.std())
    metrics["dynamic_range"] = dynamic_range(motion.energy)
    metrics["energy_corr"] = energy_correlation(
        motion.energy,
        analysis.frame["rms"].to_numpy(dtype=float),
        analysis.frame["time"].to_numpy(dtype=float),
        fps,
    )
    response_corr, response_mean = beat_response(
        motion.joints, music_beats, beat_energy, fps
    )
    metrics["beat_response_corr"] = response_corr
    metrics["beat_response_mean"] = response_mean
    metrics["slots"] = int(len(motion.plan))
    metrics["unique_clips"] = int(motion.plan["clip_id"].nunique()) if len(motion.plan) else 0
    metrics["clip_variety"] = (
        float(motion.plan["clip_id"].nunique() / len(motion.plan)) if len(motion.plan) else 0.0
    )
    metrics["duration"] = float(motion.nodes.shape[0] / fps)
    metrics["fps"] = int(fps)
    metrics["run"] = run_dir.name
    metrics["mode"] = str(motion.plan["mode"].iloc[0]) if "mode" in motion.plan else ""

    paths.save_json(run_dir / "metrics.json", metrics)
    return metrics


def compare_modes(metrics_a: dict, metrics_b: dict) -> dict:
    """Paired difference helper used by the ablation table."""
    keys = ("bas", "beat_f1", "energy_corr", "beat_response_corr", "dynamic_range")
    return {
        key: float(metrics_a.get(key, 0.0) - metrics_b.get(key, 0.0))
        for key in keys
    }


def wilcoxon_pairs(a: list[float], b: list[float]) -> dict:
    """Wilcoxon signed-rank test with graceful fallback."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    usable = np.isfinite(a) & np.isfinite(b)
    a, b = a[usable], b[usable]

    if len(a) < 5 or np.allclose(a, b):
        return {"n": int(len(a)), "statistic": None, "p_value": None}

    try:
        result = stats.wilcoxon(a, b)
        return {
            "n": int(len(a)),
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
        }
    except ValueError:
        return {"n": int(len(a)), "statistic": None, "p_value": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one generated run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    metrics = evaluate_run(Path(args.run), Path(args.features))
    print(f"Run             : {metrics['run']}")
    print(f"BAS (Gaussian)  : {metrics['bas']:.3f}")
    print(f"Beat F1         : {metrics['beat_f1']:.3f}")
    print(f"Energy corr     : {metrics['energy_corr']:.3f}")
    print(f"Beat response   : {metrics['beat_response_corr']:.3f}")
    print(f"Dynamic range   : {metrics['dynamic_range']:.2f}")
    print(f"Clip variety    : {metrics['clip_variety']:.2f}")


if __name__ == "__main__":
    main()
