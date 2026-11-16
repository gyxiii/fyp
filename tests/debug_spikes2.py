from pathlib import Path

import numpy as np

run = Path(r"C:\Users\gyixu\Documents\deepseek-harness\default-workspace\CAT404_music_animation_starter\outputs\runs\smoke_multilevel")
data = np.load(run / "motion.npz")
root = data["root"]
nodes = data["nodes"]
fps = int(data["fps"])

for center in (37, 130):
    print(f"\n=== around frame {center} ({center/fps:.2f}s) ===")
    for f in range(center - 6, center + 7):
        if 0 <= f < len(root) - 1:
            d = root[f + 1] - root[f]
            print(
                f"  f{f:4d} root=({root[f,0]:+.3f},{root[f,1]:+.3f},{root[f,2]:.3f}) "
                f"d=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) "
                f"ankleL_z={nodes[f,16,2]:+.3f}"
            )
