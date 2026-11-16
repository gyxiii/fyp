"""Static preview of an assembled motion (quality check before rendering).

Draws eight equally spaced poses from a ``motion.npz`` plus the per-frame
motion-energy curve.  Useful to verify that joints are connected, that the
character stays on the floor, and that intensity follows the music.

    python src/preview_motion.py --run outputs/runs/smoke_multilevel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import kinematics as kin
    from .motion_blend import Motion
except ImportError:  # pragma: no cover - script execution
    import kinematics as kin
    from motion_blend import Motion


EDGES = [(name, kin.NODE_INDEX[parent], kin.NODE_INDEX[child]) for name, parent, child in kin.BONES]


def draw_skeleton(ax, nodes: np.ndarray, title: str = "", color: str = "#2b2d42") -> None:
    for _, start, end in EDGES:
        ax.plot(
            [nodes[start, 0], nodes[end, 0]],
            [nodes[start, 1], nodes[end, 1]],
            [nodes[start, 2], nodes[end, 2]],
            color=color,
            linewidth=2.0,
        )
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=8, color="#e63946", depthshade=False)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=12, azim=-65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static preview of a motion.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    run_dir = Path(args.run)
    motion = Motion.load(run_dir)
    nodes = motion.nodes
    total = nodes.shape[0]

    picks = np.linspace(0, total - 1, args.frames).astype(int)
    columns = 4
    rows = int(np.ceil(args.frames / columns))

    fig = plt.figure(figsize=(4 * columns, 3.4 * rows + 2.4))
    grid = fig.add_gridspec(
        rows + 1, columns,
        height_ratios=[1.0] * rows + [0.85],
    )

    for index, frame in enumerate(picks):
        ax = fig.add_subplot(
            grid[index // columns, index % columns], projection="3d"
        )
        draw_skeleton(ax, nodes[frame], f"frame {frame} ({frame / motion.fps:.2f}s)")

    ax = fig.add_subplot(grid[rows, :])
    ax.plot(np.arange(total) / motion.fps, motion.energy, color="#457b9d", linewidth=0.8)
    for row in motion.plan.itertuples():
        ax.axvline(row.start_time, color="#e9c46a", linewidth=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean joint speed (m/frame)")
    ax.set_title("motion energy (yellow = slot boundaries)")

    fig.tight_layout()
    output = run_dir / "motion_preview.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Saved {output}")

    # Basic sanity report.
    print(f"Frames      : {total}")
    print(f"Duration    : {total / motion.fps:.2f} s")
    print(f"Root height : {motion.root[:, 2].min():.3f} - {motion.root[:, 2].max():.3f} m")
    print(f"Foot height : {nodes[:, [16, 17, 20, 21], 2].min():.3f} m (min)")
    print(f"Energy      : {motion.energy.mean():.5f} (mean), {motion.energy.max():.5f} (max)")


if __name__ == "__main__":
    main()
