"""Musical feature extraction and structure analysis (proposal section 5.2).

The module deliberately keeps the feature set small and interpretable:

    beat        -> when a movement happens
    tempo       -> how fast the movement is performed
    RMS energy  -> how strong the movement is
    onset       -> how sharp a transition is
    structure   -> which family of motion is selected

Spectral / chroma / MFCC features are extracted as well, but they are used for
*analysis only* (structure segmentation).  They are never mapped to animation
parameters, which keeps every animation decision traceable to a named feature.

Outputs (one folder per song, ``data/features/<song>/``):

    features_frame.csv   frame-level features at the animation frame rate
    features_beat.csv    beat-level features (the mapping input)
    structure.json       section boundaries, labels and energy ranks
    summary.json         tempo, duration, normalisation statistics
    feature_plot.png     visual overview used in the report

Run directly:

    python src/music_features.py data/audio/test_song.wav
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

try:  # allow "python src/music_features.py" as well as "python -m src..."
    from . import paths
except ImportError:  # pragma: no cover - script execution
    import paths


# ----------------------------------------------------------------------------
# Normalisation helpers (shared with the mapping module)
# ----------------------------------------------------------------------------

def robust_normalize(
    values,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
) -> np.ndarray:
    """Scale values to [0, 1] using robust percentiles.

    Percentile scaling keeps a single loud transient from flattening the rest
    of the song, which matters when energy drives motion intensity.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))

    if high - low < 1e-8:
        return np.full(values.shape, 0.5, dtype=float)

    return np.clip((values - low) / (high - low), 0.0, 1.0)


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------

@dataclass
class MusicAnalysis:
    """Everything the mapping stage needs to know about one song."""

    audio_path: Path
    duration: float
    sample_rate: int
    fps: int
    hop_length: int
    tempo_bpm: float
    beat_times: np.ndarray
    frame: pd.DataFrame
    beats: pd.DataFrame
    sections: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    # -- persistence -----------------------------------------------------
    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        self.frame.to_csv(out_dir / "features_frame.csv", index=False)
        self.beats.to_csv(out_dir / "features_beat.csv", index=False)
        paths.save_json(out_dir / "structure.json", self.sections)
        paths.save_json(out_dir / "summary.json", self.summary)

        beat_only = self.beats[[
            "beat_time",
            "bpm_local",
            "energy_norm",
            "onset_norm",
        ]] if len(self.beats) else pd.DataFrame()
        if len(beat_only):
            beat_only.to_csv(out_dir / "beats.csv", index=False)

        return out_dir

    @classmethod
    def load(cls, out_dir: Path, audio_path: Path | None = None) -> "MusicAnalysis":
        out_dir = Path(out_dir)
        summary = paths.load_json(out_dir / "summary.json", {}) or {}
        sections = paths.load_json(out_dir / "structure.json", []) or []
        frame = pd.read_csv(out_dir / "features_frame.csv")
        beats = pd.read_csv(out_dir / "features_beat.csv")

        return cls(
            audio_path=Path(audio_path or summary.get("audio_path", "")),
            duration=float(summary.get("duration", 0.0)),
            sample_rate=int(summary.get("sample_rate", paths.SAMPLE_RATE)),
            fps=int(summary.get("fps", paths.FPS)),
            hop_length=int(summary.get("hop_length", 0)),
            tempo_bpm=float(summary.get("tempo_bpm", 0.0)),
            beat_times=beats["beat_time"].to_numpy(dtype=float),
            frame=frame,
            beats=beats,
            sections=sections,
            summary=summary,
        )


# ----------------------------------------------------------------------------
# Frame-level features
# ----------------------------------------------------------------------------

def _band_ratio(
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Energy fraction inside [low_hz, high_hz) for every STFT frame."""
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    band = magnitude[mask].sum(axis=0)
    total = magnitude.sum(axis=0) + 1e-9
    return band / total


def compute_frame_features(
    y: np.ndarray,
    sr: int,
    fps: int = 30,
    n_fft: int = 2048,
) -> pd.DataFrame:
    """Frame-level features on a grid that matches the animation FPS.

    ``hop_length = sr / fps`` makes frame ``i`` land exactly on ``i / fps``
    seconds, so music and animation frames share one timeline.
    """
    hop_length = int(round(sr / fps))
    hop_length = max(hop_length, 1)

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    power = stft**2
    db = librosa.power_to_db(power, ref=np.max)

    rms = librosa.feature.rms(S=stft, frame_length=n_fft)[0]
    onset = librosa.onset.onset_strength(S=db, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]

    # Spectral flux: how quickly the spectrum changes, a useful "sharpness"
    # signal that complements the onset envelope.
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])
    flux = flux / (np.max(flux) + 1e-9)

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_ratio = _band_ratio(stft, frequencies, 20.0, 250.0)
    mid_ratio = _band_ratio(stft, frequencies, 250.0, 2000.0)
    high_ratio = _band_ratio(stft, frequencies, 2000.0, sr / 2.0)

    chroma = librosa.feature.chroma_stft(S=power, sr=sr)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=13)

    frame_count = min(
        len(rms), len(onset), len(centroid), len(bandwidth),
        len(rolloff), len(flatness), len(flux), chroma.shape[1], mfcc.shape[1],
    )

    times = librosa.frames_to_time(
        np.arange(frame_count), sr=sr, hop_length=hop_length
    )

    frame = pd.DataFrame({
        "time": times,
        "frame": np.arange(frame_count),
        "rms": rms[:frame_count],
        "onset": onset[:frame_count],
        "centroid": centroid[:frame_count],
        "bandwidth": bandwidth[:frame_count],
        "rolloff": rolloff[:frame_count],
        "flatness": flatness[:frame_count],
        "flux": flux[:frame_count],
        "low_ratio": low_ratio[:frame_count],
        "mid_ratio": mid_ratio[:frame_count],
        "high_ratio": high_ratio[:frame_count],
    })

    # Keep the analysis-only descriptors separate; they are returned through
    # the caller so they can feed structure segmentation without inflating the
    # mapping input.
    analysis_only = {
        "chroma": chroma[:, :frame_count],
        "mfcc": mfcc[:, :frame_count],
    }
    return frame, analysis_only


def _tempo_curve(
    onset: np.ndarray,
    sr: int,
    hop_length: int,
    frame_count: int,
) -> np.ndarray:
    """Per-frame tempo estimate; falls back to a constant curve."""
    try:
        curve = librosa.feature.tempo(
            onset_envelope=onset,
            sr=sr,
            hop_length=hop_length,
            aggregate=None,
        )
        curve = np.asarray(curve, dtype=float).reshape(-1)
        if curve.size == 1:
            return np.full(frame_count, float(curve[0]))
        if curve.size < frame_count:
            return np.pad(curve, (0, frame_count - curve.size), mode="edge")
        return curve[:frame_count]
    except Exception:
        return np.full(frame_count, float("nan"))


# ----------------------------------------------------------------------------
# Beat tracking
# ----------------------------------------------------------------------------

def track_beats(
    y: np.ndarray,
    sr: int,
    fps: int = 30,
) -> tuple[float, np.ndarray]:
    """Return (tempo_bpm, beat_times).  Falls back to a metronome grid."""
    hop_length = int(round(sr / fps))
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop_length, units="frames"
    )
    tempo_value = float(np.asarray(tempo, dtype=float).reshape(-1)[0])
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )

    duration = len(y) / sr
    if beat_times.size == 0:
        period = 60.0 / max(tempo_value, 40.0)
        beat_times = np.arange(0.0, duration, period)

    if beat_times.size and beat_times[0] > 1.5:
        # Extend the grid backwards so the opening bar is not left empty.
        period = float(np.median(np.diff(beat_times))) if beat_times.size > 1 else 0.5
        pre = np.arange(beat_times[0] - period, 0.0, -period)[::-1]
        beat_times = np.concatenate([pre, beat_times])

    beat_times = beat_times[(beat_times >= 0.0) & (beat_times < duration)]
    return tempo_value, beat_times


def estimate_downbeats(
    beat_times: np.ndarray,
    frame: pd.DataFrame,
    bar_beats: int = 4,
) -> tuple[int, np.ndarray]:
    """Estimate the 4/4 downbeat phase from low-band energy.

    Returns ``(phase, bar_position)`` where beat ``i`` belongs to bar position
    ``(i - phase) % bar_beats``.
    """
    if beat_times.size < bar_beats * 2:
        return 0, np.zeros(beat_times.size, dtype=int)

    low = np.interp(beat_times, frame["time"], frame["low_ratio"] * frame["rms"])
    best_phase, best_score = 0, -1.0

    for phase in range(bar_beats):
        indices = np.arange(phase, beat_times.size, bar_beats)
        if indices.size == 0:
            continue
        score = float(np.mean(low[indices]))
        if score > best_score:
            best_phase, best_score = phase, score

    positions = (np.arange(beat_times.size) - best_phase) % bar_beats
    return best_phase, positions


# ----------------------------------------------------------------------------
# Structure segmentation
# ----------------------------------------------------------------------------

def _novelty_curve(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Checkerboard-kernel novelty curve for a self-similarity matrix.

    The kernel compares the blocks before/after a candidate boundary.  A
    boundary is novel when the two blocks are dissimilar (negative diagonal
    quadrants) while the blocks themselves are self-similar.
    """
    half = kernel_size // 2
    sigma = max(half / 2.0, 1.0)
    axis = np.arange(-half, half)
    aa, bb = np.meshgrid(axis, axis, indexing="ij")
    kernel = np.sign(aa) * np.sign(bb) * np.exp(
        -(aa**2 + bb**2) / (2.0 * sigma**2)
    )

    padded = np.pad(ssm, half, mode="edge")
    count = ssm.shape[0]
    windows = np.stack(
        [padded[i:i + kernel_size, i:i + kernel_size] for i in range(count)]
    )
    novelty = np.einsum("ab,iab->i", kernel, windows)
    return novelty


def segment_structure(
    beat_times: np.ndarray,
    beat_frames: np.ndarray,
    chroma: np.ndarray,
    mfcc: np.ndarray,
    frame: pd.DataFrame,
    min_section_beats: int = 8,
    kernel_beats: int = 16,
    threshold: float = 0.6,
) -> list[dict]:
    """Beat-synchronous structure segmentation (verse/chorus-like sections).

    The implementation follows the classic self-similarity + checkerboard
    kernel approach: beat-synchronous chroma and MFCC are compared with a
    cosine similarity matrix, a checkerboard kernel produces a novelty curve,
    and the strongest peaks become section boundaries.
    """
    if beat_times.size < min_section_beats * 2:
        return [_make_section(0.0, float(beat_times[-1]) if beat_times.size else 0.0, 0, len(beat_times), frame, beat_times)]

    n_beats = beat_times.size
    sync_kwargs = {"aggregate": np.median}

    # Beat-synchronous features (drop the last beat frame: sync ranges).
    beat_frames = np.clip(beat_frames, 0, chroma.shape[1] - 1)
    chroma_beats = librosa.util.sync(chroma, beat_frames, **sync_kwargs)
    mfcc_beats = librosa.util.sync(mfcc[1:8], beat_frames, **sync_kwargs)

    # Trim to a common number of beats.
    n = min(chroma_beats.shape[1], mfcc_beats.shape[1], n_beats)
    if n < min_section_beats * 2:
        return [_make_section(0.0, float(beat_times[-1]), 0, n, frame, beat_times)]

    features = np.vstack([
        chroma_beats[:, :n],
        mfcc_beats[:, :n],
    ]).T.astype(float)

    features -= features.mean(axis=0, keepdims=True)
    features /= (features.std(axis=0, keepdims=True) + 1e-9)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
    normalized = features / norms

    ssm = normalized @ normalized.T
    novelty = _novelty_curve(ssm, kernel_beats)
    novelty = novelty - novelty.min()
    novelty = novelty / (novelty.max() + 1e-9)

    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=kernel_beats // 2,
        post_max=kernel_beats // 2,
        pre_avg=kernel_beats,
        post_avg=kernel_beats,
        delta=threshold * 0.25,
        wait=min_section_beats,
    )
    boundaries = sorted({int(p) for p in peaks if min_section_beats <= p <= n - min_section_beats})

    # Merge sections that are too short.
    merged: list[int] = []
    for boundary in boundaries:
        if not merged or boundary - merged[-1] >= min_section_beats:
            merged.append(boundary)
    boundaries = merged

    edges = [0] + boundaries + [n]
    sections = []
    for index in range(len(edges) - 1):
        start_beat, end_beat = edges[index], edges[index + 1]
        start_time = float(beat_times[start_beat])
        end_time = (
            float(beat_times[end_beat]) if end_beat < n_beats else float(frame["time"].iloc[-1])
        )
        sections.append(
            _make_section(start_time, end_time, start_beat, end_beat, frame, beat_times)
        )

    # Energy ranking -> motion class (0 calm, 1 mid, 2 energetic).
    energies = np.array([s["mean_energy"] for s in sections])
    order = np.argsort(energies)
    if len(sections) == 1:
        ranks = [1]
    elif len(sections) == 2:
        ranks = [0, 2]
    else:
        ranks = np.zeros(len(sections), dtype=int)
        ranks[order[0]] = 0
        ranks[order[-1]] = 2
        ranks[order[1:-1]] = 1

    labels = "ABCDEFGHIJ"
    for index, section in enumerate(sections):
        section["energy_rank"] = int(ranks[index])
        section["label"] = labels[index % len(labels)]
        section["mean_onset"] = float(
            np.mean(np.interp(
                beat_times[section["start_beat"]:section["end_beat"]],
                frame["time"], frame["onset"],
            ))
        ) if section["end_beat"] > section["start_beat"] else 0.0

    return sections


def _make_section(
    start_time: float,
    end_time: float,
    start_beat: int,
    end_beat: int,
    frame: pd.DataFrame,
    beat_times: np.ndarray,
) -> dict:
    mask = (frame["time"] >= start_time) & (frame["time"] < end_time)
    subset = frame[mask]
    mean_energy = float(subset["rms"].mean()) if len(subset) else 0.0
    return {
        "start_time": float(start_time),
        "end_time": float(end_time),
        "start_beat": int(start_beat),
        "end_beat": int(end_beat),
        "duration": float(max(end_time - start_time, 0.0)),
        "mean_energy": mean_energy,
        "energy_rank": 1,
        "label": "A",
        "mean_onset": 0.0,
    }


def section_class_at(sections: list[dict], time_sec: float) -> tuple[int, str]:
    """Return (energy_rank, label) of the section containing ``time_sec``."""
    for section in sections:
        if section["start_time"] <= time_sec < section["end_time"]:
            return int(section.get("energy_rank", 1)), str(section.get("label", "A"))
    if sections:
        last = sections[-1]
        return int(last.get("energy_rank", 1)), str(last.get("label", "A"))
    return 1, "A"


# ----------------------------------------------------------------------------
# Beat-level table
# ----------------------------------------------------------------------------

def build_beat_table(
    beat_times: np.ndarray,
    bar_position: np.ndarray,
    frame: pd.DataFrame,
    tempo_curve: np.ndarray,
    sections: list[dict],
) -> pd.DataFrame:
    """One row per beat: the direct input of the mapping engine."""
    if beat_times.size == 0:
        return pd.DataFrame()

    def at(name: str) -> np.ndarray:
        return np.interp(beat_times, frame["time"].to_numpy(), frame[name].to_numpy())

    ioi = np.diff(beat_times, append=beat_times[-1] + np.median(np.diff(beat_times)))
    ioi = np.clip(ioi, 0.05, 4.0)
    bpm_local = 60.0 / ioi

    if np.all(np.isnan(tempo_curve)):
        tempo_at_beat = bpm_local
    else:
        tempo_at_beat = np.interp(
            beat_times, frame["time"].to_numpy(), np.nan_to_num(tempo_curve, nan=np.nanmedian(tempo_curve))
        )

    energy = at("rms")
    onset = at("onset")
    classes = [section_class_at(sections, float(t)) for t in beat_times]

    beats = pd.DataFrame({
        "beat_index": np.arange(beat_times.size),
        "beat_time": beat_times,
        "bar_position": bar_position.astype(int),
        "is_downbeat": (bar_position == 0).astype(int),
        "ioi": ioi,
        "bpm_local": bpm_local,
        "bpm_curve": tempo_at_beat,
        "rms": energy,
        "onset": onset,
        "centroid": at("centroid"),
        "flux": at("flux"),
        "low_ratio": at("low_ratio"),
        "section_id": np.arange(beat_times.size),
        "section_class": [c[0] for c in classes],
        "section_label": [c[1] for c in classes],
    })

    # Add the section id by index for readability.
    section_ids = []
    for time_sec in beat_times:
        match = 0
        for index, section in enumerate(sections):
            if section["start_time"] <= time_sec < section["end_time"]:
                match = index
                break
        section_ids.append(match)
    beats["section_id"] = section_ids

    beats["energy_norm"] = robust_normalize(beats["rms"])
    beats["onset_norm"] = robust_normalize(beats["onset"])
    beats["brightness_norm"] = robust_normalize(beats["centroid"])
    beats["attack_norm"] = robust_normalize(beats["flux"])

    return beats


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def plot_features(
    analysis: MusicAnalysis,
    out_path: Path,
    y: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = analysis.frame
    beats = analysis.beats
    times = frame["time"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    if y is not None:
        axes[0].plot(np.arange(len(y)) / analysis.sample_rate, y, linewidth=0.4, color="#557")
    axes[0].plot(times, frame["rms"] * 3, color="#e07a5f", linewidth=0.9, label="RMS x3")
    axes[0].set_ylabel("waveform / RMS")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(times, frame["onset"], color="#3d5a80", linewidth=0.8)
    axes[1].set_ylabel("onset strength")

    axes[2].plot(times, frame["centroid"], color="#81b29a", linewidth=0.8)
    axes[2].set_ylabel("centroid (Hz)")

    if len(beats):
        axes[3].vlines(beats["beat_time"], 0.0, beats["energy_norm"], color="#98c1d9", linewidth=0.8, label="beat energy")
        axes[3].plot(beats["beat_time"], beats["onset_norm"], "o-", ms=2, color="#ee6c4d", linewidth=0.7, label="beat onset")
        for section in analysis.sections:
            axes[3].axvspan(section["start_time"], section["end_time"], alpha=0.08, color="#293241")
        for section in analysis.sections:
            axes[3].text(
                (section["start_time"] + section["end_time"]) / 2, 1.02,
                f"{section['label']} (cls {section['energy_rank']})",
                ha="center", va="bottom", fontsize=8,
            )
    axes[3].set_ylabel("beat features")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_ylim(-0.05, 1.18)

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{Path(analysis.audio_path).name}  |  {analysis.tempo_bpm:.1f} BPM  |  "
        f"{len(beats)} beats  |  {len(analysis.sections)} sections"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def analyse_music(
    audio_path: Path,
    out_dir: Path | None = None,
    fps: int = 30,
    sample_rate: int = 22050,
    make_plot: bool = True,
    save: bool = True,
) -> MusicAnalysis:
    """Analyse one audio file and optionally persist every artefact."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("The audio file is empty.")

    duration = float(len(y) / sr)
    frame, analysis_only = compute_frame_features(y, sr, fps=fps)
    frame_count = len(frame)
    hop_length = int(round(sr / fps))

    tempo_curve = _tempo_curve(
        frame["onset"].to_numpy(), sr, hop_length, frame_count
    )

    tempo_bpm, beat_times = track_beats(y, sr, fps=fps)
    if beat_times.size < 2:
        raise ValueError("Beat tracking found too few beats for this song.")

    # Map beat times to frame indices for beat-synchronous analysis.
    beat_frames = np.clip(
        np.round(beat_times * sr / hop_length).astype(int), 0, frame_count - 1
    )

    phase, bar_position = estimate_downbeats(beat_times, frame)
    sections = segment_structure(
        beat_times,
        beat_frames,
        analysis_only["chroma"],
        analysis_only["mfcc"],
        frame,
    )
    beats = build_beat_table(beat_times, bar_position, frame, tempo_curve, sections)

    summary = {
        "audio_path": str(audio_path),
        "audio_name": audio_path.name,
        "duration": duration,
        "sample_rate": sr,
        "fps": fps,
        "hop_length": hop_length,
        "frame_count": frame_count,
        "tempo_bpm": tempo_bpm,
        "beat_count": int(beat_times.size),
        "downbeat_phase": int(phase),
        "bar_beats": 4,
        "section_count": len(sections),
        "sections": [
            {"label": s["label"], "start": s["start_time"], "end": s["end_time"], "class": s["energy_rank"]}
            for s in sections
        ],
        "norm": {
            "low_pct": 10.0,
            "high_pct": 90.0,
            "rms_range": [float(beats["rms"].min()), float(beats["rms"].max())],
            "onset_range": [float(beats["onset"].min()), float(beats["onset"].max())],
        },
    }

    analysis = MusicAnalysis(
        audio_path=audio_path,
        duration=duration,
        sample_rate=sr,
        fps=fps,
        hop_length=hop_length,
        tempo_bpm=tempo_bpm,
        beat_times=beat_times,
        frame=frame,
        beats=beats,
        sections=sections,
        summary=summary,
    )

    if save:
        target = Path(out_dir) if out_dir else paths.feature_dir(audio_path)
        target.mkdir(parents=True, exist_ok=True)
        analysis.save(target)
        if make_plot:
            plot_features(analysis, target / "feature_plot.png", y=y)

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract musical features and structure.")
    parser.add_argument("audio", nargs="?", default=str(paths.AUDIO_DIR / "test_song.wav"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=paths.FPS)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    analysis = analyse_music(
        Path(args.audio),
        out_dir=Path(args.out) if args.out else None,
        fps=args.fps,
        make_plot=not args.no_plot,
    )

    print(f"Song       : {analysis.audio_path.name}")
    print(f"Duration   : {analysis.duration:.2f} s")
    print(f"Tempo      : {analysis.tempo_bpm:.2f} BPM")
    print(f"Beats      : {len(analysis.beats)}")
    print(f"Sections   : {len(analysis.sections)}")
    for section in analysis.sections:
        print(
            f"  {section['label']}: {section['start_time']:6.2f}-{section['end_time']:6.2f}s "
            f"class={section['energy_rank']} energy={section['mean_energy']:.4f}"
        )
    print(f"Output     : {paths.feature_dir(analysis.audio_path)}")


if __name__ == "__main__":
    main()
