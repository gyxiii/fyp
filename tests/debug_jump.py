from pathlib import Path

import numpy as np

run = Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter\outputs\runs\smoke_multilevel")
data = np.load(run / "motion.npz")
nodes = data["nodes"]
directions = data["directions"]
fps = int(data["fps"])

delta = np.linalg.norm(np.diff(nodes, axis=0), axis=-1)  # (F-1, 21)
frame = int(np.argmax(delta.max(axis=1)))
joint = int(np.argmax(delta[frame]))
print(f"max jump: frame {frame} ({frame/fps:.2f}s) node {joint}, delta {delta[frame, joint]:.3f} m")

import sys
sys.path.insert(0, str(Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter\src")))
import kinematics as kin

print("node name:", kin.NODE_NAMES[joint])
for f in range(frame - 3, frame + 4):
    print(f"  f{f}: node={nodes[f, joint]} dir_angle_changes={np.degrees(np.arccos(np.clip(np.sum(directions[f]*directions[f-1],axis=-1),-1,1)))[joint]:.1f} deg")

# Per-frame max joint delta statistics
print("\nmax joint delta per frame, top 10:")
top = np.argsort(delta.max(axis=1))[-10:][::-1]
for f in top:
    j = int(np.argmax(delta[f]))
    print(f"  f{f:4d} {f/fps:5.2f}s node {kin.NODE_NAMES[j]:>10s} delta={delta[f,j]:.3f}")
