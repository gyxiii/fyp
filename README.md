# CAT404 — Music-Driven 3D Character Animation

An interactive music-driven 3D character animation system that maps **multi-level
musical features** (beat, tempo, energy, onset, structure) onto a searchable 3D
motion library and blends the retrieved clips into a continuous character
animation rendered in Blender.

This is the implementation of the CAT404W proposal *"An Interactive Music-Driven
3D Character Animation System Using Multi-Level Musical Features"*.

## What the system does

```
WAV -> features -> mapping rules -> motion library -> blend -> Blender -> metrics
```

- **Music analysis** – Librosa features on a 30 FPS grid, beat table, tempo
  curve, RMS energy, onset strength, and novelty-based section segmentation.
- **Motion library** – 1 408 AIST++ choreographies are cut into **16 737**
  two-second clips, each described by 9 interpretable motion descriptors
  (energy, arm/leg activity, root speed, bounce, extent, rhythmicity,
  internal BPM, turning).
- **Mapping** – explicit, editable rules: beat → *when*, tempo → *how fast*,
  energy → *how strong*, onset → *transition sharpness*, structure → *motion
  family*. Six conditions are supported: `multi_level`, `beat_only` and the four
  feature ablations.
- **Blending** – time warping, spherical interpolation (Slerp) in direction
  space, per-bone intensity gain, DTW transition alignment, root continuity,
  stage anchoring, and a ground pass that keeps feet on the floor.
- **Rendering** – a rigged 21-bone mannequin in Blender, retargeted per frame,
  rendered to MP4 with the soundtrack (optional `.blend` and FBX export).
- **Evaluation** – BAS (Gaussian), beat F1, motion–music energy correlation,
  per-beat response correlation, dynamic range and clip variety, plus a
  controlled comparison / ablation runner and a Streamlit A/B rating form.

## Requirements

- Python 3.11+ virtual environment with the packages in `requirements.txt`
- Blender 5.x (default path `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`,
  override with `CAT404_BLENDER`)
- AIST++ `keypoints3d` folder (not distributed with this repository)

## Quick start (Windows PowerShell)

```powershell
cd CAT404_music_animation_starter
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Only if the AIST++ data lives outside .\data
$env:CAT404_AIST_ROOT = "D:\datasets\aistplusplus\keypoints3d"

# 1. Build the motion library once (~3 minutes)
python src\motion_library.py

# 2. Full pipeline for one song (analysis -> plan -> motion -> metrics)
python src\pipeline.py data\audio\test_song.wav --mode multi_level

# 3. Controlled experiment (4 songs x 6 conditions)
python src\experiment.py

# 4. Interactive studio
streamlit run app.py
```

Add `--render` to `src\pipeline.py` to also render the MP4 with Blender, or use
the "Motion & render" tab in the interface.

## Project layout

| Path | Role |
|---|---|
| `src/paths.py` | path resolution and data-root overrides |
| `src/music_features.py` | feature extraction, beat table, section segmentation |
| `src/motion_library.py` | AIST++ → motion library (npz + index) |
| `src/kinematics.py` | 21-node skeleton, direction Slerp, gain, forward kinematics |
| `src/mapping.py` | multi-level mapping, six conditions, clip retrieval |
| `src/motion_blend.py` | assembly, blending, DTW, ground correction |
| `src/beat_alignment.py` | BAS / F1 / energy / beat-response metrics |
| `src/experiment.py` | controlled comparison and ablation report |
| `src/pipeline.py` | end-to-end runner (optionally with Blender render) |
| `src/preview_motion.py` | static pose preview for quick QA |
| `src/make_test_variants.py` | tempo and structure test songs |
| `blender/render_character.py` | rig, retarget, render MP4 / FBX |
| `config/mapping_rules.json` | editable mapping weights |
| `app.py` | Streamlit interface (analysis, mapping, render, evaluation) |
| `docs/` | development log, experiment report, progress report |

## Outputs

Every run writes a self-describing folder:

```
outputs/runs/<song_mode>/
    plan.csv                 one row per motion slot (clip, rate, gain, blend)
    mapping_beats.csv        musical features used by the mapping
    mapping_explanation.txt  human-readable decision trace
    motion.npz               root, bone directions, lengths, nodes, joints
    motion_preview.png       eight sampled poses + energy curve
    metrics.json             BAS, beat F1, energy/beat-response correlations
    run.json                 provenance, timings, active rules
    character.mp4            optional Blender render with audio
```

## Tests

```powershell
python tests\test_app_smoke.py     # headless Streamlit render check
```

## Documentation

- `docs/DEVELOPMENT_LOG.md` — detailed iteration log, design decisions, issues
- `docs/EXPERIMENT_REPORT.md` — generated metrics tables and ablation results
- `docs/PROGRESS_REPORT.docx` — stage progress report (Chinese, with figures)

## Notes

- The motion library and the feature cache are generated artefacts; they can be
  rebuilt with `src/motion_library.py` and `src/pipeline.py`.
- On machines where the Librosa/Numba cache directory is read-only, set
  `NUMBA_CACHE_DIR` to a writable folder before running the pipeline.
## Interface

`app.py` is a Streamlit studio with five tabs:

1. **Music analysis** — waveform and feature plot, beat table, detected sections.
2. **Mapping plan** — every phrase-slot decision, with a written explanation and
   CSV / JSON export.
3. **Motion & render** — assemble and blend the clips, pose preview, Blender MP4.
4. **Evaluation** — beat-only vs multi-level comparison and a blind A/B form.
5. **Documentation** — the five mapping rules and the artifacts of a run.

Every run writes `plan.csv`, `mapping_explanation.txt`, `motion.npz`,
`metrics.json` and `run.json` into `outputs/runs/<run>/`, so a reported number
can always be traced back to the decisions that produced it.

## Data and licence

The `data/` folder is **not** part of this repository: the motion library is
derived from AIST++ (about 1 GB) and its licence does not allow redistribution.
`data/README.md` explains how to rebuild it locally. The code in this
repository is the author's own work.
