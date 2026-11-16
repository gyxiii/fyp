from pathlib import Path

import numpy as np
import pandas as pd

run = Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter\outputs\runs\smoke_multilevel")
data = np.load(run / "motion.npz")
joints = data["joints"]
root = data["root"]
nodes = data["nodes"]
fps = int(data["fps"])

speed = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(axis=1)
top = np.argsort(speed)[-12:][::-1]
print("top speed frames (frame, time, speed, root delta, max node delta):")
for f in top:
    root_delta = np.linalg.norm(root[f + 1] - root[f])
    node_delta = np.linalg.norm(nodes[f + 1] - nodes[f], axis=-1).max()
    print(f"  {f:4d} {f/fps:6.2f}s speed={speed[f]:.4f} root={root_delta:.4f} node={node_delta:.4f}")

print("\nroot range", root.min(0), root.max(0))
print("root z frame 0..5", root[:6, 2])
print("energy median", np.median(speed))

plan = pd.read_csv(run / "plan.csv")
print("\nplan start/end frames around problem frames:")
for f in top[:6]:
    t = f / fps
    nearby = plan[(plan.start_time - 0.2 < t) & (plan.end_time + 0.2 > t)]
    print(f"  t={t:.2f}: " + ", ".join(
        f"slot {int(r.slot)} [{r.start_time:.2f},{r.end_time:.2f}] clip {int(r.clip_id)} blend {r.blend_seconds:.2f}"
        for r in nearby.itertuples()
    ))
