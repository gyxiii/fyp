# Data directory (not included in this repository)

The motion library and the audio features are **derived from AIST++**, whose
licence does not permit redistribution, and they are far too large for Git
(about 1 GB). They are therefore excluded from version control.

To rebuild them:

1. Download the AIST++ `keypoints3d` folder and place it at
   `data/aistplusplus/keypoints3d/` (or point `CAT404_AIST_ROOT` at it).
2. Put one or more `.wav` files in `data/audio/`.
3. Build the clip library once:

   ```powershell
   python src\motion_library.py
   ```

   This writes `data/library/joints.npy`, `root.npy`, `motion_index.csv` and
   the per-clip descriptors.

4. Features are generated automatically on the first run of a song and cached
   in `data/features/<song>/`.

Expected layout after the build:

```
data/
  aistplusplus/keypoints3d/*.pkl   # AIST++ source data
  audio/*.wav                      # songs to animate
  features/<song>/features.csv     # cached analysis (30 FPS)
  library/joints.npy               # (clips, frames, 17, 3)
  library/root.npy
  library/motion_index.csv
  library/motion_descriptors.csv
```
