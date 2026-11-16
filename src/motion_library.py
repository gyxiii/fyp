"""Build a searchable motion library from AIST++ 3D keypoints (proposal 5.4/5.5).

AIST++ ships 1408 choreography clips (24 joints, 10 street-dance genres).  This
module converts them into short, overlapping motion segments and computes a
small set of interpretable descriptors per segment.  The descriptor set is what
the mapping stage retrieves against:

    energy        overall movement amount (how much the body travels)
    arm_activity  share of motion in the arms
    leg_activity  share of motion in the legs
    root_speed    how fast the dancer travels across the floor
    bounce        vertical range of the hips
    extent        spatial spread of the pose
    rhythmicity   periodicity of the speed signal (how "beatable" the clip is)
    turn          total body rotation

Outputs (``data/library/``):

    joints.npy        (N, T, 17, 3) float16, root-centred joint positions
    root.npy          (N, T, 3)     float32, root trajectory (metres, Z up)
    bone_lengths.npy  (N, 21)       float32, per-clip bone lengths
    rest_joints.npy   (17, 3)       float32, median rest pose
    motion_index.csv  one row per clip (descriptors + provenance)
    library_summary.json

Run:

    python src/motion_library.py                # all sequences
    python src/motion_library.py --limit 40     # quick test
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import kinematics as kin
    from . import paths
except ImportError:  # pragma: no cover - script execution
    import kinematics as kin
    import paths


GENRES = {
    "BR": "Break",
    "PO": "Pop",
    "LO": "Lock",
    "MH": "Middle Hip-hop",
    "LH": "LA Hip-hop",
    "HO": "House",
    "WA": "Waacking",
    "KR": "Krump",
    "JS": "Jazz",
    "JB": "Street Jazz",
}

FILE_PATTERN = re.compile(
    r"^g(?P<genre>[A-Z]{2})_s(?P<sub>[A-Z]{2})_cAll_d(?P<choreo>\d+)_"
    r"m(?P<music>[A-Z]{2}\d+)_ch(?P<chunk>\d+)\.pkl$"
)

SOURCE_FPS = 60.0
TARGET_FPS = 30
UNIT_SCALE = 0.01  # AIST++ keypoints are in centimetres


# ---------------------------------------------------------------------------
# Loading and conversion
# ---------------------------------------------------------------------------

def load_keypoints(file_path: Path) -> np.ndarray:
    with open(file_path, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")

    key = "keypoints3d_optim" if "keypoints3d_optim" in data else "keypoints3d"
    array = np.asarray(data[key], dtype=np.float32)

    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError(f"Unexpected keypoint shape in {file_path.name}: {array.shape}")

    return array


def convert_points(raw: np.ndarray) -> np.ndarray:
    """Source (X, Y-up, Z) -> Blender-friendly (X, Y, Z-up) in metres."""
    points = np.asarray(raw, dtype=np.float32)[:, :, [0, 2, 1]]
    return points * UNIT_SCALE


# ---------------------------------------------------------------------------
# Per-sequence feature curves
# ---------------------------------------------------------------------------

def sequence_curves(local: np.ndarray, root: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame motion quantities used to describe every clip window."""
    velocity = np.diff(local, axis=0)                 # (T-1, 17, 3)
    joint_speed = np.linalg.norm(velocity, axis=-1)   # (T-1, 17)

    arm = joint_speed[:, [7, 8, 9, 10]].mean(axis=1)
    leg = joint_speed[:, [13, 14, 15, 16]].mean(axis=1)
    upper = joint_speed[:, 5:11].mean(axis=1)
    lower = joint_speed[:, 11:17].mean(axis=1)

    root_velocity = np.diff(root, axis=0)
    root_speed = np.linalg.norm(root_velocity[:, :2], axis=-1)

    total_speed = joint_speed.mean(axis=1)
    total = total_speed + root_speed

    # Body yaw from the hip axis, unwrapped to make total turning meaningful.
    hip_lateral = local[:, 12] - local[:, 11]
    yaw = np.unwrap(np.arctan2(hip_lateral[:, 1], hip_lateral[:, 0]))

    return {
        "joint_speed": joint_speed,
        "arm": arm,
        "leg": leg,
        "upper": upper,
        "lower": lower,
        "root_speed": root_speed,
        "total_speed": total_speed,
        "total": total,
        "yaw": yaw,
        "root_height": root[:, 2],
    }


def window_sums(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Mean of ``values`` over each window (values length T-1 frames)."""
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    ends = np.minimum(starts + length, values.size)
    starts = np.minimum(starts, values.size)
    counts = np.maximum(ends - starts, 1)
    return (cumulative[ends] - cumulative[starts]) / counts


def rhythmicity(speed: np.ndarray, fps: int = TARGET_FPS) -> tuple[float, float]:
    """Periodicity of the speed signal.

    Returns ``(strength, internal_bpm)`` where ``strength`` is the peak of the
    normalised autocorrelation in the 0.3-1.2 s lag range and ``internal_bpm``
    is the tempo implied by that lag.  The internal tempo lets the mapping
    choose motion whose natural rhythm matches the music tempo.
    """
    signal = np.asarray(speed, dtype=float)
    if signal.size < fps:
        return 0.0, 120.0

    signal = signal - signal.mean()
    energy = float(np.dot(signal, signal))
    if energy < 1e-12:
        return 0.0, 120.0

    spectrum = np.fft.rfft(signal, n=2 * signal.size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum))[: signal.size]
    autocorr /= energy

    low = max(int(0.30 * fps), 2)
    high = min(int(1.20 * fps), signal.size - 1)
    if high <= low:
        return 0.0, 120.0

    window = autocorr[low:high]
    best = int(np.argmax(window)) + low
    strength = float(np.clip(autocorr[best], 0.0, 1.0))
    internal_bpm = float(np.clip(60.0 * fps / max(best, 1), 50.0, 220.0))
    return strength, internal_bpm


# ---------------------------------------------------------------------------
# Library builder
# ---------------------------------------------------------------------------

def build_library(
    output_dir: Path | None = None,
    clip_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    limit: int | None = None,
    min_energy: float = 1e-4,
    verbose: bool = True,
) -> dict:
    output_dir = Path(output_dir or paths.LIBRARY_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(paths.AIST_DIR.glob("*.pkl"))
    if limit:
        # Spread the sample across genres instead of taking the first N files.
        step = max(1, len(files) // limit)
        files = files[::step][:limit]

    if not files:
        raise FileNotFoundError(
            f"No AIST++ keypoint files found in {paths.AIST_DIR}"
        )

    clip_frames = int(round(clip_seconds * TARGET_FPS))
    hop_frames = int(round(hop_seconds * TARGET_FPS))

    all_joints: list[np.ndarray] = []
    all_roots: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    rows: list[dict] = []
    skipped = 0
    rest_samples: list[np.ndarray] = []

    for file_index, file_path in enumerate(files):
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            skipped += 1
            continue

        meta = match.groupdict()
        try:
            raw = load_keypoints(file_path)
        except Exception as error:  # corrupted file: skip, keep building
            if verbose:
                print(f"  ! skipping {file_path.name}: {error}")
            skipped += 1
            continue

        # 60 FPS -> 30 FPS
        points = convert_points(raw[::2])
        frames = points.shape[0]
        if frames < clip_frames // 2:
            skipped += 1
            continue

        # Ground the sequence so the lowest foot touches Z = 0.
        ground = float(np.percentile(points[:, [15, 16], 2], 2.0))
        points[:, :, 2] -= ground

        root = 0.5 * (points[:, 11] + points[:, 12])
        local = points - root[:, None, :]

        curves = sequence_curves(local, root)
        starts = np.arange(0, max(frames - clip_frames + 1, 1), hop_frames)

        for start in starts:
            end = start + clip_frames
            if end > frames:
                break

            window = slice(start, end)
            energy = float(window_sums(curves["joint_speed"].mean(axis=1), np.array([start]), clip_frames)[0])
            if energy < min_energy:
                continue

            arm = float(window_sums(curves["arm"], np.array([start]), clip_frames)[0])
            leg = float(window_sums(curves["leg"], np.array([start]), clip_frames)[0])
            upper = float(window_sums(curves["upper"], np.array([start]), clip_frames)[0])
            lower = float(window_sums(curves["lower"], np.array([start]), clip_frames)[0])
            root_speed = float(window_sums(curves["root_speed"], np.array([start]), clip_frames)[0])

            segment_local = local[window]
            segment_root = root[window]

            bounce = float(np.ptp(curves["root_height"][window]))
            yaw_delta = float(np.sum(np.abs(np.diff(curves["yaw"][window])))) if clip_frames > 2 else 0.0
            spread = float(np.mean(np.linalg.norm(segment_local.std(axis=0), axis=-1)))
            speed_signal = curves["total_speed"][window]
            rhythm, internal_bpm = rhythmicity(speed_signal, TARGET_FPS)

            # Bone lengths for this clip (median over frames reduces jitter).
            nodes = kin.nodes_from_joints(segment_local)
            lengths = np.array([
                np.median(np.linalg.norm(
                    nodes[:, kin.NODE_INDEX[child]] - nodes[:, kin.NODE_INDEX[parent]],
                    axis=-1,
                ))
                for _, parent, child in kin.BONES
            ], dtype=np.float32)

            all_joints.append(segment_local.astype(np.float16))
            all_roots.append(segment_root.astype(np.float32))
            all_lengths.append(lengths)
            rest_samples.append(segment_local[0].astype(np.float32))

            rows.append({
                "seq_id": file_index,
                "genre": GENRES.get(meta["genre"], meta["genre"]),
                "genre_code": meta["genre"],
                "subset": meta["sub"],
                "choreo": meta["choreo"],
                "music": meta["music"],
                "chunk": int(meta["chunk"]),
                "source_file": file_path.name,
                "start_frame": int(start),
                "start_time": float(start / TARGET_FPS),
                "clip_frames": int(clip_frames),
                "energy": energy,
                "arm_activity": arm,
                "leg_activity": leg,
                "upper_activity": upper,
                "lower_activity": lower,
                "root_speed": root_speed,
                "bounce": bounce,
                "extent": spread,
                "rhythmicity": rhythm,
                "internal_bpm": internal_bpm,
                "turn": yaw_delta,
            })

        if verbose and (file_index + 1) % 50 == 0:
            print(f"  processed {file_index + 1}/{len(files)} files, {len(rows)} clips")

    if not rows:
        raise RuntimeError("The motion library is empty; check the AIST++ data folder.")

    joints = np.stack(all_joints).astype(np.float16)
    roots = np.stack(all_roots).astype(np.float32)
    lengths = np.stack(all_lengths).astype(np.float32)
    rest_stack = np.stack(rest_samples)

    index = pd.DataFrame(rows)
    index.insert(0, "clip_id", np.arange(len(index)))

    # Robust normalisation for retrieval.
    for column in ("energy", "arm_activity", "leg_activity", "bounce", "extent", "root_speed", "turn"):
        index[f"{column}_n"] = _robust(index[column])
    index["rhythmicity_n"] = index["rhythmicity"].clip(0.0, 1.0)
    index["tempo_n"] = _robust(index["internal_bpm"], 5.0, 95.0)

    # Energy tertiles give the structure layer three motion families.  Ranking
    # is used instead of the clipped normalised value so the classes stay
    # balanced across the whole library.
    rank = index["energy"].rank(pct=True)
    index["energy_class"] = np.clip((rank * 3).astype(int), 0, 2)

    # Rest pose: median of the calmest 10 % of clip first frames.  A calm pose
    # is a much better neutral reference than a random dance frame.
    calm = index["energy"].rank(pct=True) <= 0.10
    rest_joints = np.median(rest_stack[calm.to_numpy()], axis=0).astype(np.float32)

    np.save(output_dir / "joints.npy", joints)
    np.save(output_dir / "root.npy", roots)
    np.save(output_dir / "bone_lengths.npy", lengths)
    np.save(output_dir / "rest_joints.npy", rest_joints)
    index.to_csv(output_dir / "motion_index.csv", index=False)

    summary = {
        "clip_count": int(len(index)),
        "sequence_count": int(index["seq_id"].nunique()),
        "clip_seconds": clip_seconds,
        "clip_frames": clip_frames,
        "hop_seconds": hop_seconds,
        "fps": TARGET_FPS,
        "genres": {
            code: {
                "name": GENRES.get(code, code),
                "clips": int((index["genre_code"] == code).sum()),
            }
            for code in sorted(index["genre_code"].unique())
        },
        "descriptors": [
            "energy", "arm_activity", "leg_activity", "root_speed",
            "bounce", "extent", "rhythmicity", "turn", "internal_bpm",
        ],
        "ranges": {
            column: [float(index[column].min()), float(index[column].max())]
            for column in (
                "energy", "bounce", "root_speed", "extent", "turn",
                "rhythmicity", "internal_bpm",
            )
        },
        "energy_class_counts": {
            str(key): int(value)
            for key, value in index["energy_class"].value_counts().sort_index().items()
        },
        "rest_joints": rest_joints.tolist(),
        "skipped_files": int(skipped),
        "data_root": str(paths.AIST_DIR),
    }
    paths.save_json(output_dir / "library_summary.json", summary)

    if verbose:
        print(f"Clips      : {len(index)}")
        print(f"Sequences  : {index['seq_id'].nunique()}")
        print(f"Genres     : {sorted(index['genre_code'].unique())}")
        print(f"Energy     : {index['energy'].min():.5f} - {index['energy'].max():.5f}")
        print(f"Output     : {output_dir}")

    return summary


def _robust(values, low_pct: float = 5.0, high_pct: float = 95.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(np.percentile(values, low_pct))
    high = float(np.percentile(values, high_pct))
    if high - low < 1e-9:
        return np.full(values.shape, 0.5)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AIST++ motion library.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    build_library(
        output_dir=Path(args.out) if args.out else None,
        clip_seconds=args.clip_seconds,
        hop_seconds=args.hop_seconds,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
