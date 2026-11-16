
from pathlib import Path
import pickle
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "aistplusplus"
    / "keypoints3d"
    / "gBR_sBM_cAll_d04_mBR0_ch01.pkl"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "aistplusplus"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Animation settings
# ============================================================

# Original data: 720 frames, usually 60 FPS.
# Display every 2nd frame to produce a 30 FPS animation.
FRAME_STEP = 2
OUTPUT_FPS = 30

SAVE_GIF = True
SAVE_MP4 = True

# Choose "keypoints3d_optim" for the optimized sequence.
# Change to "keypoints3d" if you want the original sequence.
DATA_KEY = "keypoints3d_optim"


# ============================================================
# 3. COCO-17 skeleton connections
# Joint indices:
# 0 nose, 1/2 eyes, 3/4 ears
# 5/6 shoulders, 7/8 elbows, 9/10 wrists
# 11/12 hips, 13/14 knees, 15/16 ankles
# ============================================================

EDGES = [
    # Face
    (0, 1), (0, 2),(1, 2),
    (1, 3), (2, 4),

    # Shoulders and torso
    (5, 6),
    (5, 11), (6, 12), (11, 12),

    # Arms
    (5, 7), (7, 9),
    (6, 8), (8, 10),

    # Legs
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# ============================================================
# 4. Load AIST++ keypoints
# ============================================================

def load_keypoints(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find the keypoint file:\n{file_path}\n"
            "Please check DATA_FILE and the filename."
        )

    # Some older pickle files may trigger a NumPy deprecation warning.
    # The warning does not necessarily mean loading has failed.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*align.*",
            category=Warning,
        )

        with open(file_path, "rb") as file:
            data = pickle.load(file, encoding="latin1")

    if DATA_KEY not in data:
        raise KeyError(
            f"'{DATA_KEY}' was not found. "
            f"Available keys: {list(data.keys())}"
        )

    keypoints = np.asarray(data[DATA_KEY], dtype=np.float32)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(
            f"Expected shape (frames, 17, 3), "
            f"but received {keypoints.shape}"
        )

    if not np.isfinite(keypoints).all():
        raise ValueError(
            "The keypoint array contains NaN or infinite values."
        )

    return keypoints


# ============================================================
# 5. Prepare coordinate system and fixed plot limits
# ============================================================

def prepare_coordinates(keypoints):
    """
    AIST++ coordinates in this file appear to use Y as vertical.

    Convert:
        original X -> display X
        original Z -> display Y (depth)
        original Y -> display Z (height)

    This keeps the animation in 3D while making Y vertical.
    """

    display = keypoints[:, :, [0, 2, 1]].copy()

    # Center the whole sequence so the figure is easier to view.
    center = (
        display.reshape(-1, 3).min(axis=0)
        + display.reshape(-1, 3).max(axis=0)
    ) / 2.0

    display -= center

    minimum = display.reshape(-1, 3).min(axis=0)
    maximum = display.reshape(-1, 3).max(axis=0)

    ranges = maximum - minimum
    max_range = max(float(np.max(ranges)), 1.0)

    # Equal spatial scale in all three directions.
    half = max_range * 0.60

    limits = []
    for axis in range(3):
        middle = (minimum[axis] + maximum[axis]) / 2.0
        limits.append((middle - half, middle + half))

    return display, limits


# ============================================================
# 6. Create the animation
# ============================================================

def main():
    print("Loading AIST++ keypoints...")
    keypoints = load_keypoints(DATA_FILE)

    total_frames = keypoints.shape[0]
    print(f"Loaded data shape: {keypoints.shape}")
    print(f"Total source frames: {total_frames}")

    keypoints, limits = prepare_coordinates(keypoints)

    # Use every second frame for a 30 FPS output.
    frame_indices = np.arange(0, total_frames, FRAME_STEP)

    duration = len(frame_indices) / OUTPUT_FPS
    print(f"Animation frames: {len(frame_indices)}")
    print(f"Output FPS: {OUTPUT_FPS}")
    print(f"Approximate duration: {duration:.2f} seconds")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_title("AIST++ 3D Human Skeleton")
    ax.set_xlabel("X")
    ax.set_ylabel("Z (depth)")
    ax.set_zlabel("Y (height)")

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])

    # The vertical direction in the plot is original Y.
    ax.view_init(elev=12, azim=-70)
    ax.set_box_aspect((1, 1, 1))

    # Keep the camera and axes fixed throughout the animation.
    ax.grid(True)

    points = ax.scatter(
        [], [], [],
        s=24,
        depthshade=False,
    )

    lines = []
    for _ in EDGES:
        line, = ax.plot(
            [], [], [],
            linewidth=2.5,
        )
        lines.append(line)

    frame_text = ax.text2D(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
    )

    def update(animation_index):
        source_index = frame_indices[animation_index]
        joints = keypoints[source_index]

        # Update all 17 joints.
        points._offsets3d = (
            joints[:, 0],
            joints[:, 1],
            joints[:, 2],
        )

        # Update every skeleton connection.
        for line, (start, end) in zip(lines, EDGES):
            line.set_data(
                [joints[start, 0], joints[end, 0]],
                [joints[start, 1], joints[end, 1]],
            )
            line.set_3d_properties(
                [joints[start, 2], joints[end, 2]]
            )

        seconds = source_index / 60.0
        frame_text.set_text(
            f"Frame: {source_index}/{total_frames - 1}"
            f"    Time: {seconds:.2f}s"
        )

        return [points, *lines, frame_text]

    # Save a separate first-frame image.
    update(0)
    first_frame_path = OUTPUT_DIR / "aist_skeleton_first_frame.png"
    fig.savefig(first_frame_path, dpi=160, bbox_inches="tight")
    print(f"Saved first frame: {first_frame_path}")

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / OUTPUT_FPS,
        blit=False,
        repeat=True,
    )

    if SAVE_GIF:
        gif_path = OUTPUT_DIR / "aist_skeleton_animation.gif"
        print("Saving GIF. This may take a few minutes...")

        animation.save(
            str(gif_path),
            writer=PillowWriter(fps=OUTPUT_FPS),
        )

        print(f"Saved GIF: {gif_path}")

    if SAVE_MP4:
        try:
            from matplotlib.animation import FFMpegWriter

            if matplotlib.animation.writers.is_available("ffmpeg"):
                mp4_path = OUTPUT_DIR / "aist_skeleton_animation.mp4"

                print("Saving MP4...")
                animation.save(
                    str(mp4_path),
                    writer=FFMpegWriter(
                        fps=OUTPUT_FPS,
                        bitrate=3000,
                    ),
                    dpi=120,
                )

                print(f"Saved MP4: {mp4_path}")
            else:
                print(
                    "FFmpeg was not detected. "
                    "The GIF is still available."
                )
                print(
                    "You can install FFmpeg later to export MP4."
                )

        except Exception as error:
            print(f"MP4 export failed: {error}")
            print("The GIF may still have been saved successfully.")

    plt.close(fig)

    print("\nFinished.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
