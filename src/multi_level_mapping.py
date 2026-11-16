from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

BEATS_PATH = PROJECT_DIR / "data" / "features" / "beats.csv"
FEATURES_PATH = PROJECT_DIR / "data" / "features" / "music_features.csv"

OUTPUT_PATH = PROJECT_DIR / "outputs" / "multi_level_events.csv"

MIN_INTENSITY = 0.35
MAX_INTENSITY = 1.50


def normalize_robust(values):
    """Normalize values to [0, 1] using the 10th and 90th percentiles."""
    values = np.asarray(values, dtype=float)

    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))

    if high - low < 1e-8:
        return np.full(len(values), 0.5)

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def main():
    if not BEATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing beat file: {BEATS_PATH}\n"
            "Run src/music_analysis.py first."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature file: {FEATURES_PATH}\n"
            "Run src/music_analysis.py first."
        )

    beats = pd.read_csv(BEATS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    required = {"time", "rms_energy", "onset_strength"}
    missing = required - set(features.columns)

    if missing:
        raise ValueError(
            f"Missing music feature columns: {sorted(missing)}"
        )

    beats["beat_time"] = pd.to_numeric(
        beats["beat_time"], errors="coerce"
    )
    beats = beats.dropna(subset=["beat_time"]).sort_values("beat_time")

    for column in ["time", "rms_energy", "onset_strength"]:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    features = (
        features.dropna(
            subset=["time", "rms_energy", "onset_strength"]
        )
        .sort_values("time")
    )

    if beats.empty or features.empty:
        raise ValueError("Beat or music feature data is empty.")

    beat_times = beats["beat_time"].to_numpy(dtype=float)
    feature_times = features["time"].to_numpy(dtype=float)

    rms_at_beats = np.interp(
        beat_times,
        feature_times,
        features["rms_energy"].to_numpy(dtype=float),
    )

    onset_at_beats = np.interp(
        beat_times,
        feature_times,
        features["onset_strength"].to_numpy(dtype=float),
    )

    # Independent feature normalization.
    rms_norm = normalize_robust(rms_at_beats)
    onset_norm = normalize_robust(onset_at_beats)

    # Multi-level mapping:
    # RMS controls the main movement amplitude.
    # Onset strength adds a separate transient emphasis.
    combined = 0.70 * rms_norm + 0.30 * onset_norm

    intensity = MIN_INTENSITY + combined * (
        MAX_INTENSITY - MIN_INTENSITY
    )

    events = pd.DataFrame({
        "time_sec": beat_times,
        "event_type": "beat",
        "intensity": intensity,
        "rms_norm": rms_norm,
        "onset_norm": onset_norm,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_PATH, index=False)

    print("Multi-level mapping completed.")
    print(f"Beat events: {len(events)}")
    print(f"Output: {OUTPUT_PATH}")
    print(
        "Intensity range: "
        f"{intensity.min():.3f} - {intensity.max():.3f}"
    )
    print("\nFirst 10 events:")
    print(events.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()