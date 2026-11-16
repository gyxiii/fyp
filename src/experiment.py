"""Controlled comparison and ablation experiment (proposal RQ1-RQ3).

Runs every mapping condition on every test song under identical conditions
(same motion library, same retrieval code, same blending, same evaluation) and
writes:

    outputs/experiments/metrics.csv        one row per (song, mode)
    outputs/experiments/summary.csv        mean/std per mode
    outputs/experiments/comparison.csv     paired deltas against beat-only
    outputs/experiments/experiment_summary.png
    docs/EXPERIMENT_REPORT.md              generated report with tables

Conditions:

    beat_only      beats drive timing only; tempo/energy/onset/structure neutral
    multi_level    all five layers active
    no_tempo       ablation: tempo layer removed
    no_energy      ablation: energy layer removed
    no_onset       ablation: onset layer removed
    no_structure   ablation: structure layer removed

Run:

    python src/experiment.py                       # all songs, all modes
    python src/experiment.py --songs test_song     # one song
    python src/experiment.py --modes multi_level beat_only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import beat_alignment, mapping, paths, pipeline
except ImportError:  # pragma: no cover - script execution
    import beat_alignment
    import mapping
    import paths
    import pipeline


CONDITIONS = ("beat_only", "multi_level", "no_tempo", "no_energy", "no_onset", "no_structure")
METRICS = ("bas", "beat_f1", "energy_corr", "beat_response_corr", "dynamic_range", "clip_variety")

LABELS = {
    "beat_only": "Beat-only",
    "multi_level": "Multi-level",
    "no_tempo": "No tempo",
    "no_energy": "No energy",
    "no_onset": "No onset",
    "no_structure": "No structure",
}


def discover_songs(pattern: str | None = None) -> list[Path]:
    files = [
        path for path in paths.audio_files()
        if path.name.startswith(("test_song", "demo_"))
    ]
    if pattern:
        files = [path for path in files if pattern in path.name]
    return sorted(files)


def run_experiment(
    songs: list[Path],
    modes: tuple[str, ...] = CONDITIONS,
    refresh_features: bool = False,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir or paths.OUTPUT_DIR / "experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.time()

    for song in songs:
        print(f"\n=== {song.name} ===", flush=True)
        for mode in modes:
            run_name = f"exp_{paths.slug(song.stem)}_{mode}"
            print(f"  {mode:>13s} ...", end="", flush=True)
            started_mode = time.time()

            try:
                metadata = pipeline.run_song(
                    song,
                    mode=mode,
                    run_name=run_name,
                    refresh_features=refresh_features,
                )
            except Exception as error:  # keep the experiment running
                print(f" FAILED: {error}", flush=True)
                rows.append({
                    "song": song.name,
                    "mode": mode,
                    "error": str(error),
                })
                continue

            metrics = metadata["metrics"]
            row = {
                "song": song.name,
                "mode": mode,
                "tempo_bpm": metadata["tempo_bpm"],
                "beats": metadata["beats"],
                "sections": metadata["sections"],
                "slots": metadata["slots"],
                "run_dir": metadata["run_dir"],
                "seconds": round(time.time() - started_mode, 1),
            }
            row.update({key: metrics.get(key) for key in METRICS})
            rows.append(row)
            print(
                f" BAS={metrics['bas']:.3f} F1={metrics['beat_f1']:.3f} "
                f"EC={metrics['energy_corr']:.3f} BR={metrics['beat_response_corr']:.3f} "
                f"({row['seconds']:.0f}s)",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\nTotal experiment time: {time.time() - started:.0f} s")

    if not table.empty and "error" not in table.columns:
        summarise(table, output_dir)
    return table


def summarise(table: pd.DataFrame, output_dir: Path) -> None:
    usable = table.dropna(subset=["bas"])
    if usable.empty:
        return

    summary = usable.groupby("mode")[list(METRICS)].agg(["mean", "std"]).round(4)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Paired comparison against the beat-only baseline.
    pivot = usable.pivot_table(index="song", columns="mode", values=list(METRICS))
    rows = []
    for mode in usable["mode"].unique():
        if mode == "beat_only":
            continue
        row = {"mode": mode}
        for metric in METRICS:
            try:
                a = pivot[(metric, mode)].to_numpy(dtype=float)
                b = pivot[(metric, "beat_only")].to_numpy(dtype=float)
            except KeyError:
                continue
            deltas = a - b
            row[f"{metric}_delta"] = float(np.nanmean(deltas))
            row[f"{metric}_wins"] = int(np.nansum(deltas > 0))
            row[f"{metric}_n"] = int(np.sum(np.isfinite(deltas)))
            if metric in ("bas", "energy_corr"):
                row[f"{metric}_p"] = beat_alignment.wilcoxon_pairs(a, b)["p_value"]
        rows.append(row)

    comparison = pd.DataFrame(rows).round(4)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    plot_summary(usable, output_dir / "experiment_summary.png")
    write_report(usable, summary, comparison, output_dir)
    print(f"\nSaved summary to {output_dir}")


def plot_summary(table: pd.DataFrame, output_path: Path) -> None:
    modes = [mode for mode in CONDITIONS if mode in set(table["mode"])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ("bas", "BAS (Gaussian)", axes[0, 0]),
        ("beat_f1", "Beat F1", axes[0, 1]),
        ("energy_corr", "Motion-music energy correlation", axes[1, 0]),
        ("beat_response_corr", "Per-beat response correlation", axes[1, 1]),
    ]

    for metric, title, ax in panels:
        means = [table[table["mode"] == mode][metric].mean() for mode in modes]
        stds = [table[table["mode"] == mode][metric].std() for mode in modes]
        colors = ["#8d99ae" if mode == "beat_only" else "#2a6f97" for mode in modes]
        ax.bar([LABELS.get(mode, mode) for mode in modes], means, yerr=stds, color=colors, capsize=4)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Beat-only vs multi-level mapping and feature ablations")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a small Markdown table (no tabulate dependency)."""
    if frame.empty:
        return "_(empty)_"

    columns = list(frame.columns)
    lines = ["| " + " | ".join(str(column) for column in columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")

    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def write_report(
    table: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    report = []
    report.append("# Controlled Experiment Report (auto-generated)\n")
    report.append(
        "This file is generated by `src/experiment.py`. Every row below was produced "
        "by the same motion library, retrieval code, blending pipeline and evaluation "
        "code; only the active mapping layers differ.\n"
    )

    report.append("## Songs\n")
    songs = table[["song", "tempo_bpm", "beats", "sections"]].drop_duplicates("song")
    report.append(md_table(songs))
    report.append("")

    report.append("## Per-song metrics\n")
    report.append(
        md_table(
            table[["song", "mode", "bas", "beat_f1", "energy_corr", "beat_response_corr",
                   "dynamic_range", "clip_variety"]].round(3)
        )
    )
    report.append("")

    report.append("## Mean by condition\n")
    report.append(md_table(summary.round(3)))
    report.append("")

    report.append("## Paired deltas against beat-only\n")
    report.append(
        "Positive deltas mean the condition scores higher than beat-only on the same songs. "
        "`wins` counts songs where the condition beat the baseline. "
        "Wilcoxon signed-rank tests need at least five pairs; with four songs the p-value is "
        "reported as `n/a` and the comparison stays descriptive.\n"
    )
    display = comparison.astype(object).where(comparison.notna(), "n/a")
    report.append(md_table(display))
    report.append("")

    report.append("## Notes\n")
    report.append(
        "- Songs are short (19-116 s) and only four conditions' songs are available, "
        "so the paired statistics are descriptive rather than confirmatory.\n"
        "- BAS uses the kinematic beat detector in `src/beat_alignment.py` (sigma = 0.15 s), "
        "not a human annotation.\n"
        "- The multi-level condition is expected to raise expressiveness (energy and beat-response "
        "correlation) more than synchronisation, because both conditions place clips on the beat grid.\n"
    )

    target = paths.DOCS_DIR / "EXPERIMENT_REPORT.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(report), encoding="utf-8")
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled mapping experiment.")
    parser.add_argument("--songs", default=None, help="substring filter on song filename")
    parser.add_argument("--modes", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--refresh-features", action="store_true")
    args = parser.parse_args()

    songs = discover_songs(args.songs)
    if not songs:
        raise FileNotFoundError("No test songs found in data/audio.")

    run_experiment(
        songs,
        modes=tuple(args.modes),
        refresh_features=args.refresh_features,
    )


if __name__ == "__main__":
    main()
