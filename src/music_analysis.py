from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
AUDIO_PATH = PROJECT_DIR / "data" / "audio" / "test_song.wav"
OUTPUT_DIR = PROJECT_DIR / "data" / "features"


def analyse_music(audio_path: Path = AUDIO_PATH, output_dir: Path = OUTPUT_DIR):
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\n"
            "Copy a WAV file to data/audio/test_song.wav first."
        )

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if y.size == 0:
        raise ValueError("The audio file is empty.")

    duration = len(y) / sr
    hop_length = 512

    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="frames"
    )
    beat_times = librosa.frames_to_time(
        beat_frames, sr=sr, hop_length=hop_length
    )

    rms = librosa.feature.rms(
        y=y, frame_length=2048, hop_length=hop_length
    )[0]
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )

    frame_count = min(len(rms), len(onset))
    times = librosa.frames_to_time(
        np.arange(frame_count), sr=sr, hop_length=hop_length
    )

    features = pd.DataFrame({
        "time": times,
        "rms_energy": rms[:frame_count],
        "onset_strength": onset[:frame_count],
    })
    features.to_csv(output_dir / "music_features.csv", index=False)
    pd.DataFrame({"beat_time": beat_times}).to_csv(
        output_dir / "beats.csv", index=False
    )

    tempo_value = float(np.asarray(tempo).reshape(-1)[0])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(np.arange(len(y)) / sr, y, linewidth=0.5)
    axes[0].set(title="Audio waveform", xlabel="Time (s)")

    axes[1].plot(times, rms[:frame_count])
    axes[1].set(title="RMS energy", xlabel="Time (s)")

    axes[2].plot(times, onset[:frame_count], label="Onset strength")
    if len(beat_times):
        beat_y = np.interp(beat_times, times, onset[:frame_count])
        axes[2].scatter(beat_times, beat_y, marker="x", label="Detected beats")
    axes[2].set(title="Onset strength and detected beats", xlabel="Time (s)")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "feature_plot.png", dpi=150)
    plt.close(fig)

    print(f"Audio: {audio_path.name}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Estimated tempo: {tempo_value:.2f} BPM")
    print(f"Detected beats: {len(beat_times)}")
    print(f"Saved features to: {output_dir}")

    return features, beat_times, tempo_value


if __name__ == "__main__":
    analyse_music()
