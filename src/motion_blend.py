"""Motion retrieval, time-warping and blending (proposal sections 5.5/5.6).

The assembled animation is stored in *direction space*: a root trajectory plus
one unit direction per bone.  This representation makes the three operations in
this module straightforward and numerically stable:

    time warping   -> spherical interpolation between source frames
    intensity      -> scaling the angle between rest and current bone direction
    blending       -> spherical interpolation between the outgoing and the
                      incoming motion over a short overlap

A final ground pass keeps feet from sinking below the floor, and a DTW-based
temporal search aligns the incoming clip with the outgoing tail before blending.

Outputs (one folder per run):

    motion.npz      root, directions, lengths, nodes, joints, energy curve
    preview.png     quick static preview of four key frames
    motion_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import kinematics as kin
    from . import paths
except ImportError:  # pragma: no cover - script execution
    import kinematics as kin
    import paths


# ---------------------------------------------------------------------------
# Library access
# ---------------------------------------------------------------------------

class ClipLibrary:
    """Random access to the motion library stored as memory-mapped arrays."""

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or paths.LIBRARY_DIR)
        self.index = pd.read_csv(self.directory / "motion_index.csv")
        self.joints = np.load(self.directory / "joints.npy", mmap_mode="r")
        self.roots = np.load(self.directory / "root.npy", mmap_mode="r")
        self.lengths = np.load(self.directory / "bone_lengths.npy", mmap_mode="r")
        self.rest_joints = np.load(self.directory / "rest_joints.npy")
        self.skeleton = kin.Skeleton(kin.nodes_from_joints(self.rest_joints))

        summary = paths.load_json(self.directory / "library_summary.json", {}) or {}
        self.fps = int(summary.get("fps", paths.FPS))
        self.summary = summary
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.index)

    def row(self, clip_id: int) -> pd.Series:
        return self.index.iloc[int(clip_id)]

    def clip(self, clip_id: int) -> dict:
        """Return a yaw-normalised clip: joints, directions, root, lengths."""
        clip_id = int(clip_id)
        if clip_id in self._cache:
            return self._cache[clip_id]

        joints = np.asarray(self.joints[clip_id], dtype=np.float32)
        root = np.asarray(self.roots[clip_id], dtype=np.float32)
        lengths = np.asarray(self.lengths[clip_id], dtype=np.float32)

        # Rotate the clip so every clip starts facing +Y.  Without this, clips
        # from different AIST++ recordings face different directions and the
        # character would spin at every transition.
        lateral = joints[0, 12] - joints[0, 11]
        facing = float(np.arctan2(lateral[1], lateral[0]) + np.pi / 2.0)
        cos_a, sin_a = np.cos(-facing), np.sin(-facing)
        rotation = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        joints = joints @ rotation.T
        root = root @ rotation.T

        nodes = kin.nodes_from_joints(joints)
        directions = self.skeleton.directions_from_nodes(nodes)

        clip = {
            "clip_id": clip_id,
            "joints": joints,
            "root": root,
            "lengths": lengths,
            "directions": directions,
        }
        if len(self._cache) < 256:
            self._cache[clip_id] = clip
        return clip


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def slerp_array(a: np.ndarray, b: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Vectorised spherical interpolation of direction arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    weight = np.asarray(t, dtype=float)
    if weight.ndim == 1:
        weight = weight[:, None]
    weight = np.broadcast_to(weight, a.shape[:-1])[..., None]

    dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)

    # Antipodal directions have no unique great-circle path.  Nudge them along
    # an arbitrary perpendicular axis so the interpolation stays continuous
    # instead of collapsing to a zero vector and flipping.
    opposite = dot < -0.9995
    if np.any(opposite):
        perpendicular = np.cross(a, np.array([0.0, 0.0, 1.0]))
        weak = np.linalg.norm(perpendicular, axis=-1, keepdims=True) < 1e-6
        perpendicular = np.where(weak, np.cross(a, np.array([1.0, 0.0, 0.0])), perpendicular)
        perpendicular = perpendicular / np.maximum(
            np.linalg.norm(perpendicular, axis=-1, keepdims=True), 1e-9
        )
        nudged = b + 1e-3 * perpendicular
        b = np.where(opposite, nudged / np.maximum(np.linalg.norm(nudged, axis=-1, keepdims=True), 1e-9), b)
        dot = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6

    mixed = np.where(
        small,
        (1.0 - weight) * a + weight * b,
        (np.sin((1.0 - weight) * theta) / np.maximum(sin_theta, 1e-9)) * a
        + (np.sin(weight * theta) / np.maximum(sin_theta, 1e-9)) * b,
    )
    return mixed / np.maximum(np.linalg.norm(mixed, axis=-1, keepdims=True), 1e-9)


def sample_clip(
    clip: dict,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a clip at fractional frame positions with spherical interpolation."""
    directions = clip["directions"]
    frames = directions.shape[0]

    # Clamp instead of wrapping: the extended tail holds the last pose, which
    # is always overwritten by the next clip at full weight.
    wrapped = np.clip(positions, 0.0, frames - 1.0)
    lower = np.floor(wrapped).astype(int)
    upper = np.minimum(lower + 1, frames - 1)
    fraction = wrapped - lower

    sampled_directions = slerp_array(directions[lower], directions[upper], fraction)

    root = clip["root"]
    root_sampled = np.stack([
        np.interp(wrapped, np.arange(frames), root[:, axis])
        for axis in range(3)
    ], axis=1)

    return sampled_directions, root_sampled.astype(np.float32), clip["lengths"]


def _smooth_directions(directions: np.ndarray, window: int = 3) -> np.ndarray:
    """Short temporal smoothing of bone directions (removes one-frame pops).

    Directions are averaged with a triangular kernel and re-normalised.  The
    window is deliberately tiny (three frames at 30 FPS = 0.1 s) so the dance
    rhythm is not softened, only single-frame tracking glitches.
    """
    if window < 2:
        return directions

    half = window // 2
    padded = np.pad(directions, ((half, half), (0, 0), (0, 0)), mode="edge")
    smoothed = np.zeros_like(directions)
    total = 0.0
    for offset in range(window):
        weight = float(window - abs(offset - half))
        smoothed += padded[offset:offset + directions.shape[0]] * weight
        total += weight
    smoothed /= total
    return smoothed / np.maximum(np.linalg.norm(smoothed, axis=-1, keepdims=True), 1e-9)


def smoothstep(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Smooth step between ``edges[0]`` and ``edges[1]``."""
    low, high = float(edges[0]), float(edges[1])
    if high - low < 1e-6:
        return np.ones_like(values)
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _limit_vertical_speed(height: np.ndarray, max_step: float = 0.020) -> np.ndarray:
    """Slew-rate limiter for the root height.

    Hip height is driven by the pose, and two clips can meet at very different
    heights.  Limiting the per-frame change keeps the transition readable
    without touching horizontal motion.
    """
    result = np.asarray(height, dtype=np.float32).copy()
    for frame in range(1, len(result)):
        delta = result[frame] - result[frame - 1]
        if delta > max_step:
            result[frame] = result[frame - 1] + max_step
        elif delta < -max_step:
            result[frame] = result[frame - 1] - max_step

    # Symmetric pass removes the drift introduced by the causal pass.
    for frame in range(len(result) - 2, -1, -1):
        delta = result[frame] - result[frame + 1]
        if delta > max_step:
            result[frame] = result[frame + 1] + max_step
        elif delta < -max_step:
            result[frame] = result[frame + 1] - max_step

    return result


# ---------------------------------------------------------------------------
# DTW alignment
# ---------------------------------------------------------------------------

def dtw_offset(previous: np.ndarray, incoming: np.ndarray) -> int:
    """Median frame offset that best aligns ``incoming`` with ``previous``.

    Both inputs are (frames, joints, 3) node sequences.  A classic dynamic time
    warping table is computed over their root-relative poses, and the median
    index difference along the optimal path becomes a small start offset.
    """
    if previous.shape[0] < 3 or incoming.shape[0] < 3:
        return 0

    a = previous - previous[:, :1, :]
    b = incoming - incoming[:, :1, :]
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)

    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)

    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    n, m = cost.shape

    accumulated = np.full((n, m), np.inf)
    accumulated[0, 0] = cost[0, 0]
    for i in range(1, n):
        accumulated[i, 0] = accumulated[i - 1, 0] + cost[i, 0]
    for j in range(1, m):
        accumulated[0, j] = accumulated[0, j - 1] + cost[0, j]
    for i in range(1, n):
        for j in range(1, m):
            accumulated[i, j] = cost[i, j] + min(
                accumulated[i - 1, j],
                accumulated[i, j - 1],
                accumulated[i - 1, j - 1],
            )

    i, j = n - 1, m - 1
    offsets = []
    while i > 0 or j > 0:
        offsets.append(j - i)
        candidates = []
        if i > 0 and j > 0:
            candidates.append((accumulated[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            candidates.append((accumulated[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((accumulated[i, j - 1], i, j - 1))
        _, i, j = min(candidates, key=lambda item: item[0])

    offset = int(np.median(offsets)) if offsets else 0
    return int(np.clip(offset, -3, 3))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class Motion:
    fps: int
    root: np.ndarray            # (F, 3)
    directions: np.ndarray      # (F, 21, 3)
    lengths: np.ndarray         # (21,)
    nodes: np.ndarray           # (F, 21, 3)
    joints: np.ndarray          # (F, 17, 3)
    energy: np.ndarray          # (F,) motion amount per frame
    plan: pd.DataFrame
    summary: dict

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "motion.npz",
            fps=self.fps,
            root=self.root.astype(np.float32),
            directions=self.directions.astype(np.float32),
            lengths=self.lengths.astype(np.float32),
            nodes=self.nodes.astype(np.float32),
            joints=self.joints.astype(np.float32),
            energy=self.energy.astype(np.float32),
        )
        self.plan.to_csv(directory / "plan.csv", index=False)
        paths.save_json(directory / "motion_summary.json", self.summary)
        return directory

    @classmethod
    def load(cls, directory: Path) -> "Motion":
        directory = Path(directory)
        data = np.load(directory / "motion.npz")
        plan_path = directory / "plan.csv"
        plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
        summary = paths.load_json(directory / "motion_summary.json", {})
        return cls(
            fps=int(data["fps"]),
            root=data["root"],
            directions=data["directions"],
            lengths=data["lengths"],
            nodes=data["nodes"],
            joints=data["joints"],
            energy=data["energy"],
            plan=plan,
            summary=summary or {},
        )


def build_motion(
    plan: pd.DataFrame,
    library: ClipLibrary,
    rules: dict,
    fps: int = 30,
    verbose: bool = True,
) -> Motion:
    """Assemble the clip plan into one continuous motion."""
    plan = plan.sort_values("start_time").reset_index(drop=True)
    if plan.empty:
        raise ValueError("The motion plan is empty.")

    duration = float(plan["end_time"].max()) + 0.5
    total_frames = int(np.ceil(duration * fps)) + 2

    skeleton = library.skeleton
    rest_dirs = skeleton.rest_directions

    directions = np.zeros((total_frames, len(kin.BONES), 3), dtype=np.float32)
    root = np.zeros((total_frames, 3), dtype=np.float32)
    filled = np.zeros(total_frames, dtype=bool)

    lengths = np.median(np.asarray(library.lengths[: min(len(library), 500)]), axis=0).astype(np.float32)

    blend_rules = rules.get("blend", {})
    use_dtw = bool(blend_rules.get("dtw", True))

    # Every clip is kept alive past its own blend window so that the *next*
    # clip always finds a fully weighted pose to blend from.  Without this the
    # first frame after a clip ends would be written unblended (a visible pop).
    max_blend_frames = int(plan["blend_seconds"].max() * fps) + 4

    previous_tail: np.ndarray | None = None

    for row_index, row in plan.iterrows():
        clip = library.clip(int(row["clip_id"]))
        start_frame = int(round(float(row["start_time"]) * fps))
        end_frame = int(round(float(row["end_time"]) * fps))
        blend_frames = max(int(round(float(row["blend_seconds"]) * fps)), 1)

        if end_frame <= start_frame:
            continue

        # The clip must cover the slot plus the transition tail.  ``rate`` is a
        # bounded tempo nuance on top of the beat-grid fit.
        own_frames = (end_frame - start_frame) + blend_frames
        cover_frames = (end_frame - start_frame) + max_blend_frames
        source_frames = clip["directions"].shape[0]
        timing_fit = source_frames / max(own_frames, 1)
        rate = float(row.get("rate", 1.0))
        speed = timing_fit * float(np.clip(rate, 0.85, 1.15))

        positions = np.arange(cover_frames, dtype=float) * speed
        slot_directions, slot_root, slot_lengths = sample_clip(clip, positions)

        gain = float(row.get("gain", 1.0))
        gain = float(np.clip(gain, 0.2, 2.0))

        # Intensity: scale every bone angle away from the rest pose.
        scaled = np.stack([
            kin.scale_direction(
                slot_directions[:, bone_index],
                rest_dirs[bone_index],
                gain,
            )
            for bone_index in range(len(kin.BONES))
        ], axis=1).astype(np.float32)

        nodes = skeleton.fk(slot_root, scaled, slot_lengths)

        # Small DTW-based alignment against the previously placed motion.
        if use_dtw and previous_tail is not None and previous_tail.shape[0] >= 3:
            head = nodes[: min(blend_frames, nodes.shape[0])]
            tail = previous_tail[-min(blend_frames, previous_tail.shape[0]):]
            offset = dtw_offset(tail, head)
            if offset:
                start_frame = max(start_frame + offset, 0)

        # Root continuity: match the XY position already accumulated.
        if 0 <= start_frame < total_frames and filled[start_frame]:
            target_xy = root[start_frame, :2]
            delta_xy = target_xy - slot_root[0, :2]
            slot_root = slot_root.copy()
            slot_root[:, :2] += delta_xy

        # Stage anchoring: AIST++ clips travel across the floor.  Keeping a
        # fraction of that travel leaves a visible weight shift while stopping
        # the character from walking out of the camera frame.
        anchor = float(blend_rules.get("root_anchor", 0.45))
        if anchor < 1.0:
            origin = slot_root[0, :2].copy()
            slot_root = slot_root.copy()
            slot_root[:, :2] = origin + (slot_root[:, :2] - origin) * anchor

        weight_profile = smoothstep(
            np.array([start_frame, start_frame + blend_frames], dtype=float),
            np.arange(start_frame, start_frame + cover_frames, dtype=float),
        )

        # A true cross-fade needs the outgoing pose as the blending origin.
        # Blending repeatedly from the accumulated value compounds the change
        # and produces a visible snap, so snapshot the region first.
        write_end = min(start_frame + cover_frames, total_frames)
        base_directions = directions[start_frame:write_end].copy()
        base_root = root[start_frame:write_end].copy()
        base_filled = filled[start_frame:write_end].copy()
        base_count = write_end - start_frame

        if verbose and row_index < 3:
            print(
                f"  slot {int(row['slot'])} clip {int(row['clip_id'])}: start={start_frame} "
                f"end={end_frame} blend={blend_frames} out={cover_frames} "
                f"prefilled={int(base_filled.sum())}/{base_count}",
                flush=True,
            )

        for local_index in range(base_count):
            frame = start_frame + local_index
            weight = float(weight_profile[local_index])
            if not base_filled[local_index]:
                directions[frame] = scaled[local_index]
                root[frame] = slot_root[local_index]
                filled[frame] = True
            else:
                directions[frame] = slerp_array(
                    base_directions[local_index][None],
                    scaled[local_index][None],
                    weight,
                )[0]
                root[frame] = (
                    (1.0 - weight) * base_root[local_index]
                    + weight * slot_root[local_index]
                )

        tail_start = max(start_frame + cover_frames - blend_frames, start_frame)
        previous_tail = nodes[max(tail_start - start_frame, 0):]

    # Fill frames that received no clip at all (e.g. before the first beat).
    if not filled[0]:
        first = int(np.argmax(filled))
        for frame in range(first):
            directions[frame] = directions[first]
            root[frame] = root[first]
            filled[frame] = True
    if not filled[-1]:
        last = int(len(filled) - 1 - np.argmax(filled[::-1]))
        for frame in range(last + 1, total_frames):
            directions[frame] = directions[last]
            root[frame] = root[last]
            filled[frame] = True

    # 1. remove single-frame tracking pops, 2. keep hip height physically
    # plausible (a clip boundary can otherwise look like a teleport), 3. keep
    # feet on the floor.
    directions = _smooth_directions(
        directions, int(blend_rules.get("smooth_frames", 3))
    )
    nodes = skeleton.fk(root, directions, lengths)
    root[:, 2] = _limit_vertical_speed(root[:, 2], max_step=0.020)
    nodes = skeleton.fk(root, directions, lengths)

    # Ground pass: never let a foot sink through the floor.
    if blend_rules.get("foot_lock", True):
        foot_indices = [
            kin.NODE_INDEX["ankle.L"], kin.NODE_INDEX["toe.L"],
            kin.NODE_INDEX["ankle.R"], kin.NODE_INDEX["toe.R"],
        ]
        foot_z = nodes[:, foot_indices, 2].min(axis=1)
        # The ankle joint sits ~7 cm above the sole, so keep it clear of the
        # floor rather than touching it.
        clearance = float(blend_rules.get("foot_clearance", 0.07))
        correction = np.maximum(clearance - foot_z, 0.0)
        if correction.max() > 1e-4:
            # Dilate before smoothing so the correction still covers the peak.
            from scipy.ndimage import maximum_filter1d, uniform_filter1d

            correction = maximum_filter1d(correction, size=5, mode="nearest")
            correction = uniform_filter1d(correction, size=3, mode="nearest")
            root[:, 2] += correction.astype(np.float32)
            nodes = skeleton.fk(root, directions, lengths)

    joints = kin.joints_from_nodes(nodes)
    velocity = np.diff(joints, axis=0)
    energy = np.concatenate([[0.0], np.linalg.norm(velocity, axis=-1).mean(axis=1)])

    duration_seconds = total_frames / fps
    summary = {
        "fps": fps,
        "frames": int(total_frames),
        "duration": float(duration_seconds),
        "slots": int(len(plan)),
        "mean_speed": float(np.mean(energy)),
        "max_speed": float(np.max(energy)),
        "mean_root_speed": float(np.mean(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1))),
        "foot_correction_max": float(np.max(np.abs(nodes[:, [16, 20], 2]))),
        "unique_clips": int(plan["clip_id"].nunique()),
        "genres": sorted(plan["clip_genre"].unique().tolist()),
    }

    if verbose:
        print(f"Frames       : {total_frames} ({duration_seconds:.2f} s @ {fps} fps)")
        print(f"Slots        : {len(plan)}")
        print(f"Mean speed   : {summary['mean_speed']:.5f} m/frame")
        print(f"Unique clips : {summary['unique_clips']}")

    return Motion(
        fps=fps,
        root=root,
        directions=directions,
        lengths=lengths,
        nodes=nodes,
        joints=joints,
        energy=energy,
        plan=plan,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a motion plan into continuous motion.")
    parser.add_argument("--run", required=True, help="Run folder containing plan.csv")
    parser.add_argument("--library", default=str(paths.LIBRARY_DIR))
    parser.add_argument("--rules", default=str(paths.RULES_PATH))
    args = parser.parse_args()

    run_dir = Path(args.run)
    plan = pd.read_csv(run_dir / "plan.csv")
    rules = paths.load_json(Path(args.rules))
    library = ClipLibrary(Path(args.library))

    motion = build_motion(plan, library, rules)
    motion.save(run_dir)
    print(f"Saved motion to {run_dir / 'motion.npz'}")


if __name__ == "__main__":
    main()

