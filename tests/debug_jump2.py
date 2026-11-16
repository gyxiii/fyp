import sys
from pathlib import Path

import numpy as np

root_dir = Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter")
sys.path.insert(0, str(root_dir / "src"))
import kinematics as kin

data = np.load(root_dir / "outputs/runs/smoke_multilevel/motion.npz")
nodes = data["nodes"]
directions = data["directions"]

for f in range(35, 41):
    print(f"\nf{f}")
    for name in ("hips", "spine_mid", "chest", "shoulder.L", "elbow.L", "wrist.L", "hand.L"):
        i = kin.NODE_INDEX[name]
        print(f"  {name:>10s} {np.round(nodes[f, i], 3)}")
    angles = np.degrees(np.arccos(np.clip(np.sum(directions[f] * directions[f-1], axis=-1), -1, 1)))
    print("  bone angle changes:", ", ".join(f"{n}:{a:.0f}" for n, a in zip(kin.BONE_NAMES, angles)))
