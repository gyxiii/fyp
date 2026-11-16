
from pathlib import Path
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_DIR / "data" / "aistplusplus" / "keypoints3d"
    / "gBR_sBM_cAll_d04_mBR0_ch01.pkl"
)
OUTPUT_DIR = PROJECT_DIR / "outputs" / "aistplusplus"

# COCO-style candidate connections; verify against the
# exact AIST++ joint definition before using for retargeting.
EDGES = [
    # Face
    (0, 1), (0, 2),
    (1, 3), (2, 4),

    # Shoulders and torso
    (5, 6),
    (5, 11), (6, 12), (11, 12),

    # Left arm
    (5, 7), (7, 9),

    # Right arm
    (6, 8), (8, 10),

    # Left leg
    (11, 13), (13, 15),

    # Right leg
    (12, 14), (14, 16),
]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with DATA_FILE.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    points = np.asarray(data["keypoints3d_optim"], dtype=float)
    frame = points[0]

    if frame.shape != (17, 3):
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    # Reorder axes for display: original Y is the vertical axis.
    # New display coordinates are (original X, original Z, original Y).
    display = frame[:, [0, 2, 1]]
    valid = np.isfinite(display).all(axis=1)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        display[valid, 0],
        display[valid, 1],
        display[valid, 2],
        s=45,
    )

    for a, b in EDGES:
        if valid[a] and valid[b]:
            ax.plot(
                [display[a, 0], display[b, 0]],
                [display[a, 1], display[b, 1]],
                [display[a, 2], display[b, 2]],
                linewidth=2,
            )

    # Equal visual scaling.
    xyz = display[valid]
    center = xyz.mean(axis=0)
    radius = max(float(np.ptp(xyz, axis=0).max()) / 2, 1.0)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("X")
    ax.set_ylabel("Depth (original Z)")
    ax.set_zlabel("Height (original Y)")
    ax.set_title("AIST++ 3D Keypoints - Y-up View")
    ax.view_init(elev=12, azim=-75)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aist_first_frame_y_up.png"
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    print("Saved:", output_path)
    print("Data shape:", points.shape)
    print("Valid joints:", valid.sum(), "/ 17")

if __name__ == "__main__":
    main()
