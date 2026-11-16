
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
BEATS_PATH = PROJECT_DIR / "data" / "features" / "beats.csv"
FEATURES_PATH = PROJECT_DIR / "data" / "features" / "music_features.csv"
OUTPUT_PATH = PROJECT_DIR / "outputs" / "animation_events.csv"

# Animation intensity range.
MIN_INTENSITY = 0.35
MAX_INTENSITY = 1.50


def main():
    if not BEATS_PATH.exists():
        raise FileNotFoundError(
            "beats.csv not found. Run `python src/music_analysis.py` first."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "music_features.csv not found. Run `python src/music_analysis.py` first."
        )

    beats = pd.read_csv(BEATS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    if "beat_time" not in beats.columns:
        raise ValueError("beats.csv must contain a 'beat_time' column.")

    required_columns = {"time", "rms_energy"}
    if not required_columns.issubset(features.columns):
        raise ValueError(
            "music_features.csv must contain 'time' and 'rms_energy' columns."
        )

    # Remove invalid rows and sort by time.
    beats = beats[["beat_time"]].copy()
    beats["beat_time"] = pd.to_numeric(
        beats["beat_time"], errors="coerce"
    )
    beats = beats.dropna().sort_values("beat_time")

    features = features[["time", "rms_energy"]].copy()
    features["time"] = pd.to_numeric(features["time"], errors="coerce")
    features["rms_energy"] = pd.to_numeric(
        features["rms_energy"], errors="coerce"
    )
    features = features.dropna().sort_values("time")

    if beats.empty:
        raise ValueError("No valid beat times were found.")

    if features.empty:
        raise ValueError("No valid RMS energy features were found.")

    # Match each beat to the RMS energy at the same time.
    beat_times = beats["beat_time"].to_numpy(dtype=float)
    feature_times = features["time"].to_numpy(dtype=float)
    rms_values = features["rms_energy"].to_numpy(dtype=float)

    beat_energy = np.interp(
        beat_times,
        feature_times,
        rms_values,
    )

    # Use percentiles to reduce the influence of extreme energy values.
    low = float(np.percentile(beat_energy, 10))
    high = float(np.percentile(beat_energy, 90))

    if high - low < 1e-8:
        normalized_energy = np.full(len(beat_energy), 0.5)
    else:
        normalized_energy = (beat_energy - low) / (high - low)
        normalized_energy = np.clip(normalized_energy, 0.0, 1.0)

    # Convert normalized audio energy to animation intensity.
    intensity = MIN_INTENSITY + normalized_energy * (
        MAX_INTENSITY - MIN_INTENSITY
    )

    events = pd.DataFrame({
        "time_sec": beat_times,
        "event_type": "beat",
        "intensity": intensity,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_PATH, index=False)

    print(f"Created {len(events)} beat events.")
    print(f"RMS energy range: {beat_energy.min():.4f} - {beat_energy.max():.4f}")
    print(
        f"Animation intensity range: "
        f"{intensity.min():.2f} - {intensity.max():.2f}"
    )
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nFirst 10 animation events:")
    print(events.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
