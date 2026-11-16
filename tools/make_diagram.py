"""Draw the pipeline diagram used in the progress report (matplotlib only)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "docs" / "pipeline_diagram.png"


def main() -> None:
    stages = [
        ("WAV", "audio input"),
        ("Music\nfeatures", "beat / tempo / energy\nonset / structure"),
        ("Mapping\nrules", "5 layers\n6 conditions"),
        ("Motion\nlibrary", "16 737 clips\n10 genres"),
        ("Blend", "slerp + DTW\nfoot lock"),
        ("Blender", "rig + render\nMP4 / FBX"),
        ("Metrics", "BAS / F1\nenergy corr."),
    ]

    fig, ax = plt.subplots(figsize=(13, 2.9))
    ax.set_xlim(0, len(stages) * 2)
    ax.set_ylim(0, 2)
    ax.axis("off")

    colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#6a4c93", "#2a9d8f"]

    for index, ((title, subtitle), color) in enumerate(zip(stages, colors)):
        x = index * 2 + 0.15
        box = plt.Rectangle(
            (x, 0.55), 1.45, 1.0, facecolor=color, alpha=0.18,
            edgecolor=color, linewidth=1.6, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + 0.72, 1.28, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#1d3557", zorder=3)
        ax.text(x + 0.72, 0.86, subtitle, ha="center", va="center", fontsize=8,
                color="#333333", zorder=3)
        if index < len(stages) - 1:
            ax.annotate("", xy=(x + 1.93, 1.05), xytext=(x + 1.62, 1.05),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))

    ax.text(
        0.15, 0.18,
        "Every stage writes its artefacts into outputs/runs/<song_mode>/ "
        "(plan.csv, mapping_explanation.txt, motion.npz, metrics.json)",
        fontsize=8, color="#555555",
    )

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
