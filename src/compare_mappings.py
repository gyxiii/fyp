from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

BASELINE_PATH = PROJECT_DIR / "outputs" / "animation_events.csv"
MULTI_PATH = PROJECT_DIR / "outputs" / "multi_level_events.csv"


def main():
    baseline = pd.read_csv(BASELINE_PATH)
    multi = pd.read_csv(MULTI_PATH)

    baseline = baseline.sort_values("time_sec").reset_index(drop=True)
    multi = multi.sort_values("time_sec").reset_index(drop=True)

    if len(baseline) != len(multi):
        raise ValueError("The two methods have different event counts.")

    if not np.allclose(
        baseline["time_sec"].to_numpy(),
        multi["time_sec"].to_numpy(),
        atol=1e-6,
    ):
        raise ValueError("Beat timestamps do not match.")

    a = baseline["intensity"].to_numpy(dtype=float)
    b = multi["intensity"].to_numpy(dtype=float)

    difference = b - a

    print("===== Mapping Comparison =====")
    print(f"Baseline events: {len(a)}")
    print(f"Multi-level events: {len(b)}")
    print(f"Same beat timestamps: Yes")
    print()
    print(f"Baseline mean intensity: {a.mean():.4f}")
    print(f"Multi-level mean intensity: {b.mean():.4f}")
    print(f"Mean absolute intensity difference: "
          f"{np.mean(np.abs(difference)):.4f}")
    print(f"Maximum absolute intensity difference: "
          f"{np.max(np.abs(difference)):.4f}")
    print(f"Events with changed intensity: "
          f"{np.sum(np.abs(difference) > 1e-6)} / {len(a)}")

    result = pd.DataFrame({
        "time_sec": baseline["time_sec"],
        "baseline_intensity": a,
        "multi_level_intensity": b,
        "intensity_difference": difference,
    })

    output_path = PROJECT_DIR / "outputs" / "mapping_comparison.csv"
    result.to_csv(output_path, index=False)

    print(f"\nDetailed comparison saved to: {output_path}")
    print("\nFirst 10 comparisons:")
    print(result.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()