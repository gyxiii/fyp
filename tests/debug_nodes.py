import sys
from pathlib import Path

import numpy as np

root_dir = Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter")
sys.path.insert(0, str(root_dir / "src"))
import kinematics as kin

data = np.load(root_dir / "outputs/runs/smoke_multilevel/motion.npz")
nodes = data["nodes"]
root = data["root"]
lengths = data["lengths"]

print("z min/max per key node:")
for name in ("hips", "knee.L", "ankle.L", "toe.L", "knee.R", "ankle.R", "toe.R", "head"):
    i = kin.NODE_INDEX[name]
    print(f"  {name:>8s}: {nodes[:, i, 2].min():+.3f} .. {nodes[:, i, 2].max():+.3f}")

print("\nbone length check (dist between nodes vs stored length):")
for i, (name, parent, child) in enumerate(kin.BONES):
    d = np.linalg.norm(nodes[:, kin.NODE_INDEX[child]] - nodes[:, kin.NODE_INDEX[parent]], axis=-1)
    print(f"  {name:>12s}: actual {d.mean():.3f} +- {d.std():.4f}  stored {lengths[i]:.3f}")

f = 317
print(f"\nframe {f}:")
for name in ("hips", "knee.L", "ankle.L", "toe.L", "knee.R", "ankle.R", "toe.R"):
    i = kin.NODE_INDEX[name]
    print(f"  {name:>8s}: {np.round(nodes[f, i], 3)}")
