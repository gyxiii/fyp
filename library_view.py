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


@functools.lru_cache(maxsize=256)
def pose_gif(
    clip_id: int,
    frames: int = 14,
    size: int = 240,
    fps: int = 14,
    genre: str = "",
    background: str = "#FFFFFF",
) -> bytes:
    """An animated, seamlessly looping preview of one clip."""
    joints = _joints()
    if joints is None:
        return b""
    clip = np.asarray(joints[int(clip_id)], dtype=np.float32)
    picks = np.linspace(0, clip.shape[0] - 1, frames, endpoint=False).astype(int)

    colour = genre_colour(genre)
    images = []
    for frame in picks:
        image = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(image)
        px, py = _project(clip, frame, size)
        _draw_pose(draw, px, py, size, colour)
        images.append(image)

    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return buffer.getvalue()



# ---------------------------------------------------------------------------
# Clay-style rendering (soft capsules + shading, painted back to front)
# ---------------------------------------------------------------------------

def _rgb(colour: str) -> tuple[int, int, int]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def _mix(rgb, target, amount: float):
    return tuple(int(round(c + (t - c) * amount)) for c, t in zip(rgb, target))


def _capsule(draw, p0, p1, radius, base, shadow, light):
    """A rounded limb with a soft shadow edge and a top-left highlight."""
    width = max(3, int(radius * 2))
    draw.line([p0, p1], fill=shadow, width=width + 3)
    for point in (p0, p1):
        draw.ellipse((point[0] - radius - 1.5, point[1] - radius - 1.5,
                      point[0] + radius + 1.5, point[1] + radius + 1.5), fill=shadow)
    draw.line([p0, p1], fill=base, width=width)
    for point in (p0, p1):
        draw.ellipse((point[0] - radius, point[1] - radius,
                      point[0] + radius, point[1] + radius), fill=base)
    dx, dy = -radius * 0.32, -radius * 0.40
    draw.line([(p0[0] + dx, p0[1] + dy), (p1[0] + dx, p1[1] + dy)],
              fill=light, width=max(2, int(radius * 0.75)))
    for point in (p0, p1):
        draw.ellipse((point[0] + dx - radius * 0.42, point[1] + dy - radius * 0.42,
                      point[0] + dx + radius * 0.42, point[1] + dy + radius * 0.42), fill=light)


def _sphere(draw, point, radius, base, shadow, light):
    draw.ellipse((point[0] - radius, point[1] - radius,
                  point[0] + radius, point[1] + radius), fill=shadow)
    inner = radius * 0.9
    draw.ellipse((point[0] - inner, point[1] - inner,
                  point[0] + inner, point[1] + inner), fill=base)
    hl = radius * 0.42
    cx, cy = point[0] - radius * 0.30, point[1] - radius * 0.38
    draw.ellipse((cx - hl, cy - hl, cx + hl, cy + hl), fill=light)


def _project_clay(clip: np.ndarray, frame: int, size: int, fill: float = 0.80):
    """Stable scale across frames, normalised by the median body height."""
    x = clip[:, :, 0].astype(np.float32)
    z = clip[:, :, 2].astype(np.float32)
    heights = z.max(axis=1) - z.min(axis=1)
    height = float(np.median(heights[heights > 1e-3])) if np.any(heights > 1e-3) else 1.0
    scale = (size * fill) / max(height, 1e-3)
    cx = float(np.median((x.max(axis=1) + x.min(axis=1)) / 2))
    cz = float(np.median((z.max(axis=1) + z.min(axis=1)) / 2))
    px = (x[frame] - cx) * scale + size / 2
    py = size / 2 - (z[frame] - cz) * scale
    return px, py


def _clay_frame(clip: np.ndarray, frame: int, size: int, genre: str, background: str):
    base = _rgb(genre_colour(genre))
    shadow = _mix(base, (25, 20, 35), 0.30)
    light = _mix(base, (255, 255, 255), 0.58)
    limb = _mix(base, (255, 255, 255), 0.46)
    limb_shadow = _mix(limb, (25, 20, 35), 0.30)
    limb_light = _mix(limb, (255, 255, 255), 0.55)

    px, py = _project_clay(clip, frame, size)
    depth = clip[frame, :, 1].astype(np.float32)

    image = Image.new("RGBA", (size, size), background)
    draw = ImageDraw.Draw(image)

    # soft ground shadow
    foot_y = float(max(py[15], py[16]))
    draw.ellipse((size * 0.33, foot_y + size * 0.015, size * 0.67, foot_y + size * 0.075),
                 fill=(120, 110, 130, 26))

    def mid(a, b):
        return ((px[a] + px[b]) / 2, (py[a] + py[b]) / 2)

    neck = mid(5, 6)
    pelvis = mid(11, 12)
    spine = ((neck[0] * 0.55 + pelvis[0] * 0.45), (neck[1] * 0.55 + pelvis[1] * 0.45))
    head_r = size * 0.082
    head_center = (neck[0] + (px[0] - neck[0]) * 0.10, neck[1] - head_r * 0.95)

    limbs = [
        (5, 7, 0.030), (6, 8, 0.030),
        (7, 9, 0.026), (8, 10, 0.026),
        (11, 13, 0.040), (12, 14, 0.040),
        (13, 15, 0.033), (14, 16, 0.033),
        (neck[0], neck[1], head_center[0], head_center[1], 0.024),
    ]

    body = []
    for item in limbs:
        if len(item) == 3:
            a, b, r = item
            body.append(((px[a], py[a]), (px[b], py[b]), r, float((depth[a] + depth[b]) / 2)))
        else:
            x0, y0, x1, y1, r = item
            body.append(((x0, y0), (x1, y1), r, 0.0))
    body.sort(key=lambda item: item[3], reverse=True)

    # torso first, then the limbs, then the head
    _capsule(draw, neck, spine, 0.072 * size, base, shadow, light)
    _capsule(draw, spine, pelvis, 0.060 * size, base, shadow, light)
    _capsule(draw, (px[5], py[5]), (px[6], py[6]), 0.050 * size, base, shadow, light)
    _capsule(draw, (px[11], py[11]), (px[12], py[12]), 0.052 * size, base, shadow, light)

    for p0, p1, r, _ in body:
        _capsule(draw, p0, p1, r * size, limb, limb_shadow, limb_light)

    # head and extremities
    head_fill = _mix(base, (255, 255, 255), 0.55)
    _sphere(draw, head_center, head_r, head_fill, _mix(head_fill, (25, 20, 35), 0.30), _mix(head_fill, (255, 255, 255), 0.55))
    _sphere(draw, (px[9], py[9]), size * 0.026, limb, limb_shadow, limb_light)
    _sphere(draw, (px[10], py[10]), size * 0.026, limb, limb_shadow, limb_light)
    _sphere(draw, (px[15], py[15]), size * 0.030, limb, limb_shadow, limb_light)
    _sphere(draw, (px[16], py[16]), size * 0.030, limb, limb_shadow, limb_light)
    return image


def clay_png(clip_id: int, frame: int | None = None, size: int = 240, genre: str = "",
             background: str = "#FFFFFF") -> bytes:
    """One clay-style frame as PNG bytes."""
    joints = _joints()
    if joints is None:
        return b""
    clip = np.asarray(joints[int(clip_id)], dtype=np.float32)
    if frame is None:
        frame = _best_frame(clip)
    frame = int(np.clip(frame, 0, clip.shape[0] - 1))
    image = _clay_frame(clip, frame, size, genre, background).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


@functools.lru_cache(maxsize=256)
def clay_gif(clip_id: int, frames: int = 16, size: int = 220, fps: int = 15,
             genre: str = "", background: str = "#FFFFFF") -> bytes:
    """A looping clay-style preview."""
    joints = _joints()
    if joints is None:
        return b""
    clip = np.asarray(joints[int(clip_id)], dtype=np.float32)
    picks = np.linspace(0, clip.shape[0] - 1, frames, endpoint=False).astype(int)
    images = [_clay_frame(clip, int(frame), size, genre, background).convert("RGB") for frame in picks]

    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return buffer.getvalue()
