import sys
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter")
sys.path.insert(0, str(root_dir / "src"))
import kinematics as kin
from motion_blend import ClipLibrary, sample_clip, slerp_array

lib = ClipLibrary()
for clip_id in (1894, 12781, 9095):
    clip = lib.clip(clip_id)
    d = clip["directions"]
    angles = np.degrees(np.arccos(np.clip(np.sum(d[1:] * d[:-1], axis=-1), -1, 1)))
    print(f"clip {clip_id}: max source angle change {angles.max():.1f} deg at frame {int(np.argmax(angles.max(axis=1)))+1}, "
          f"mean {angles.mean():.1f}")
    per_bone = angles.max(axis=0)
    worse = np.argsort(per_bone)[-3:][::-1]
    print("   worst bones:", [(kin.BONE_NAMES[i], round(float(per_bone[i]), 1)) for i in worse])

# sampled version
clip = lib.clip(12781)
positions = np.arange(0, 42, dtype=float) * (60 / 42) * 0.94
dirs, root, lengths = sample_clip(clip, positions)
angles = np.degrees(np.arccos(np.clip(np.sum(dirs[1:] * dirs[:-1], axis=-1), -1, 1)))
print("\nsampled clip 12781 per-frame max angle change:")
for i in range(3, 10):
    print(f"  out frame {i}: {angles[i-1].max():.1f} deg, forearm.L {angles[i-1, kin.BONE_INDEX['forearm.L']]:.1f}")

plan = pd.read_csv(root_dir / "outputs/runs/smoke_multilevel/plan.csv")
print("\nplan rows 0-2:")
print(plan.loc[:2, ["slot", "start_time", "end_time", "clip_id", "rate", "blend_seconds"]].to_string(index=False))
