"""Render motion-library clips as skeleton thumbnails (PIL, no extra deps)."""

from __future__ import annotations

import functools
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = PROJECT_DIR / "data" / "library"
JOINTS_PATH = LIBRARY_DIR / "joints.npy"

# COCO-17 skeleton edges used by the AIST++ keypoints.
BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

GENRE_COLOURS = {
    "Break": "#6C5CE7",
    "Pop": "#00B8D9",
    "Lock": "#E17055",
    "Middle Hip-hop": "#0984E3",
    "LA Hip-hop": "#00B894",
    "House": "#C79518",
    "Waacking": "#E84393",
    "Krump": "#D63031",
    "Jazz": "#8C7AE6",
    "Street Jazz": "#16A085",
}

DEFAULT_COLOUR = "#6C5CE7"


def genre_colour(genre: str) -> str:
    return GENRE_COLOURS.get(genre, DEFAULT_COLOUR)


@functools.lru_cache(maxsize=1)
def _joints():
    if not JOINTS_PATH.exists():
        return None
    return np.load(JOINTS_PATH, mmap_mode="r")


def _best_frame(clip: np.ndarray) -> int:
    """Frame with the largest spread — usually the most readable pose."""
    x = clip[:, :, 0]
    z = clip[:, :, 2]
    spread = (x.max(axis=1) - x.min(axis=1)) + (z.max(axis=1) - z.min(axis=1))
    return int(np.argmax(spread))


def _project(clip: np.ndarray, frame: int, size: int, margin: float = 0.16):
    """Project the whole clip's bounding box so strip frames share one scale."""
    x = clip[:, :, 0].astype(np.float32)
    z = clip[:, :, 2].astype(np.float32)
    cx = float((x.min() + x.max()) / 2)
    cz = float((z.min() + z.max()) / 2)
    span = max(float(x.max() - x.min()), float(z.max() - z.min()), 1e-3)
    scale = (size * (1 - 2 * margin)) / span
    px = (x[frame] - cx) * scale + size / 2
    py = size / 2 - (z[frame] - cz) * scale
    return px, py


def _draw_pose(draw: ImageDraw.ImageDraw, px, py, size: int, colour: str, offset: float = 0.0):
    width = max(2, size // 55)
    radius = max(2, size // 85)
    for a, b in BONES:
        draw.line((px[a] + offset, py[a], px[b] + offset, py[b]), fill=colour, width=width)
    for index in range(px.shape[0]):
        draw.ellipse(
            (px[index] + offset - radius, py[index] - radius,
             px[index] + offset + radius, py[index] + radius),
            fill="#1B1F3B",
        )


def pose_png(clip_id: int, frame: int | None = None, size: int = 240, genre: str = "") -> bytes:
    """One skeleton frame as transparent PNG bytes."""
    joints = _joints()
    if joints is None:
        return b""
    clip = np.asarray(joints[int(clip_id)], dtype=np.float32)
    if frame is None:
        frame = _best_frame(clip)
    frame = int(np.clip(frame, 0, clip.shape[0] - 1))

    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    px, py = _project(clip, frame, size)
    _draw_pose(draw, px, py, size, genre_colour(genre))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def strip_png(clip_id: int, frames: int = 6, cell: int = 150, genre: str = "") -> bytes:
    """A filmstrip of evenly spaced frames."""
    joints = _joints()
    if joints is None:
        return b""
    clip = np.asarray(joints[int(clip_id)], dtype=np.float32)
    picks = np.linspace(0, clip.shape[0] - 1, frames).astype(int)

    image = Image.new("RGBA", (cell * frames, cell), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    colour = genre_colour(genre)
    for column, frame in enumerate(picks):
        px, py = _project(clip, frame, cell)
        _draw_pose(draw, px, py, cell, colour, offset=column * cell)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
