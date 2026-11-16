"""Create small test songs used by the ablation experiment.

The project ships one 23 s test song.  A controlled comparison needs several
songs with different tempi and at least one song with clear sections.  These
variants are generated locally from the shipped song, so no additional audio is
required and the experiment stays reproducible:

    demo_100bpm.wav     time-stretched to ~102 BPM
    demo_145bpm.wav     time-stretched to ~144 BPM
    demo_structure.wav  quiet / loud / quiet arrangement (~46 s) for structure tests

Run:

    python src/make_test_variants.py
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    from . import paths
except ImportError:  # pragma: no cover - script execution
    import paths


def write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(str(path), (audio / peak * 0.95).astype(np.float32), sr)
    print(f"  {path.name}: {len(audio) / sr:.2f} s @ {sr} Hz")


def main() -> None:
    source = paths.AUDIO_DIR / "test_song.wav"
    if not source.exists():
        raise FileNotFoundError(f"Missing base song: {source}")

    y, sr = librosa.load(str(source), sr=22050, mono=True)
    print(f"Base song: {source.name} ({len(y) / sr:.2f} s)")

    print("Tempo variants:")
    write(paths.AUDIO_DIR / "demo_100bpm.wav", librosa.effects.time_stretch(y, rate=0.85), sr)
    write(paths.AUDIO_DIR / "demo_145bpm.wav", librosa.effects.time_stretch(y, rate=1.20), sr)

    print("Structure variant:")
    quiet = y * 0.35
    loud = y
    mid = y * 0.65
    structure = np.concatenate([quiet, mid, loud, mid, quiet])
    write(paths.AUDIO_DIR / "demo_structure.wav", structure, sr)


if __name__ == "__main__":
    main()
