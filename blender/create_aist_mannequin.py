
import bpy
import pickle
import math
from pathlib import Path

from mathutils import Vector
import numpy as np


# ============================================================
# 1. Paths: preserve the existing skeleton project
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "aistplusplus"
    / "keypoints3d"
    / "gBR_sBM_cAll_d04_mBR0_ch01.pkl"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "aistplusplus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLEND_FILE = OUTPUT_DIR / "aist_mannequin.blend"

DATA_KEY = "keypoints3d_optim"
FRAME_STEP = 2
OUTPUT_FPS = 30

# The original data appears to use Y as the vertical axis.
# Convert source (X, Y, Z) into Blender (X, Y, Z-up):
# Blender X = source X
# Blender Y = source Z
# Blender Z = source Y
UNIT_SCALE = 0.01

# Smooth mannequin part widths in Blender units.
HEAD_SCALE = (0.105, 0.105, 0.145)
NECK_RADIUS = 0.045
TORSO_RADIUS = 0.105
ARM_RADIUS = 0.040
FOREARM_RADIUS = 0.032
LEG_RADIUS = 0.055
LOWER_LEG_RADIUS = 0.042
HAND_RADIUS = 0.032
FOOT_RADIUS = 0.040


# ============================================================
# 2. Load keypoints
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

with open(DATA_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

if DATA_KEY not in data:
    raise KeyError(
        f"Missing '{DATA_KEY}'. Available keys: {list(data.keys())}"
    )

raw = np.asarray(data[DATA_KEY], dtype=np.float32)

if raw.ndim != 3 or raw.shape[1:] != (17, 3):
    raise ValueError(f"Unexpected keypoint shape: {raw.shape}")

if not np.isfinite(raw).all():
    raise ValueError("Keypoint data contains invalid numeric values.")

# Use every second source frame: 720 -> 360 frames.
source_indices = list(range(0, raw.shape[0], FRAME_STEP))

# Convert coordinates to Blender's Z-up convention.
points = raw[:, :, [0, 2, 1]].copy()

# Center the complete sequence, not each individual frame.
bounds_min = points.reshape(-1, 3).min(axis=0)
bounds_max = points.reshape(-1, 3).max(axis=0)
center = (bounds_min + bounds_max) / 2.0
points = (points - center) * UNIT_SCALE

print("Loaded:", raw.shape)
print("Frames to animate:", len(source_indices))


# ============================================================
# 3. Clear current scene
# ============================================================

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

for datablocks in (
    bpy.data.meshes,
    bpy.data.curves,
    bpy.data.cameras,
    bpy.data.lights,
):
    for block in list(datablocks):
        if block.users == 0:
            datablocks.remove(block)


# ============================================================
# 4. Materials
# ============================================================

def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55

    return mat


skin_mat = make_material("Mannequin Warm Gray", (0.63, 0.67, 0.72))
joint_mat = make_material("Joint Material", (0.10, 0.48, 0.95))
floor_mat = make_material("Dark Floor", (0.035, 0.045, 0.065))


# ============================================================
# 5. Create smooth ellipsoid parts
# ============================================================

def create_ellipsoid(name, material, scale):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=16,
        ring_count=12,
        radius=1.0,
        location=(0, 0, 0),
    )

    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    obj.scale = scale

    for polygon in obj.data.polygons:
        polygon.use_smooth = True

    return obj


def animate_point(obj, position, frame):
    obj.location = Vector(position)
    obj.keyframe_insert(data_path="location", frame=frame)


def animate_segment(obj, start, end, radius, frame):
    start = Vector(start)
    end = Vector(end)

    direction = end - start
    length = direction.length

    obj.location = (start + end) / 2.0

    if length > 1e-6:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
        obj.scale = (radius, radius, length / 2.0 + radius)
    else:
        obj.scale = (radius, radius, radius)

    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    obj.keyframe_insert(data_path="scale", frame=frame)


# ============================================================
# 6. Define human body parts using COCO-17 joints
# ============================================================
#
# 0 nose, 1/2 eyes, 3/4 ears
# 5/6 shoulders, 7/8 elbows, 9/10 wrists
# 11/12 hips, 13/14 knees, 15/16 ankles
#
# We intentionally do not connect the ears to the shoulders.
# The head is a simple ellipsoid centered around the face points.
#
# Body parts are visual segments driven by keypoints, not a
# deforming skinned mesh / Armature rig.

segments = [
    # Neck: center of face to midpoint of shoulders
    ("Neck", "neck", NECK_RADIUS),

    # Torso
    ("Torso", "torso", TORSO_RADIUS),

    # Upper arms and forearms
    ("UpperArm_L", (5, 7), ARM_RADIUS),
    ("Forearm_L", (7, 9), FOREARM_RADIUS),
    ("UpperArm_R", (6, 8), ARM_RADIUS),
    ("Forearm_R", (8, 10), FOREARM_RADIUS),

    # Thighs and lower legs
    ("Thigh_L", (11, 13), LEG_RADIUS),
    ("LowerLeg_L", (13, 15), LOWER_LEG_RADIUS),
    ("Thigh_R", (12, 14), LEG_RADIUS),
    ("LowerLeg_R", (14, 16), LOWER_LEG_RADIUS),

    # Hands and feet
    ("Hand_L", "hand_l", HAND_RADIUS),
    ("Hand_R", "hand_r", HAND_RADIUS),
    ("Foot_L", "foot_l", FOOT_RADIUS),
    ("Foot_R", "foot_r", FOOT_RADIUS),
]


# Create objects once; animate them across all frames.
head = create_ellipsoid("Head", skin_mat, HEAD_SCALE)

joint_objects = {}
for i in range(17):
    joint_objects[i] = create_ellipsoid(
        f"Joint_{i:02d}",
        joint_mat,
        (0.025, 0.025, 0.025),
    )

segment_objects = {}
for name, _, radius in segments:
    segment_objects[name] = create_ellipsoid(
        name,
        skin_mat,
        (radius, radius, 0.1),
    )


# ============================================================
# 7. Animate mannequin from each source frame
# ============================================================

def midpoint(a, b):
    return (Vector(a) + Vector(b)) / 2.0


for output_index, source_index in enumerate(source_indices):
    frame = output_index + 1
    p = points[source_index]

    # Head center from face keypoints.
    # COCO-17 has no top-of-head or chin point.
    face_center = p[[0, 1, 2, 3, 4]].mean(axis=0)
    shoulders = midpoint(p[5], p[6])
    hips = midpoint(p[11], p[12])

    # Head follows face position.
    animate_point(head, face_center, frame)
    head.keyframe_insert(data_path="scale", frame=frame)

    # Show the 17 keypoints as small blue joints.
    for i, obj in joint_objects.items():
        animate_point(obj, p[i], frame)

    for name, kind, radius in segments:
        obj = segment_objects[name]

        if kind == "neck":
            start, end = face_center, shoulders

        elif kind == "torso":
            start, end = shoulders, hips

        elif kind == "hand_l":
            start, end = p[9], p[9] + (p[9] - p[7]) * 0.18

        elif kind == "hand_r":
            start, end = p[10], p[10] + (p[10] - p[8]) * 0.18

        elif kind == "foot_l":
            start, end = p[15], p[15] + Vector((0.075, 0, -0.015))

        elif kind == "foot_r":
            start, end = p[16], p[16] + Vector((0.075, 0, -0.015))

        else:
            start_index, end_index = kind
            start, end = p[start_index], p[end_index]

        animate_segment(obj, start, end, radius, frame)

    if frame % 60 == 1:
        print(f"Animated frame {frame}/{len(source_indices)}")


# Linear interpolation avoids overshoot between sampled poses.
# Blender 5.2 uses layered Actions in many contexts, so avoid
# relying on the removed legacy action.fcurves interface here.
# Blender's default interpolation is acceptable for this first pass.


# ============================================================
# 8. Floor, camera, and lights
# ============================================================

# Place the floor below the lowest point of the centered sequence.
floor_z = float(points[:, :, 2].min()) - 0.10

bpy.ops.mesh.primitive_plane_add(
    size=8,
    location=(0, 0, floor_z),
)
floor = bpy.context.object
floor.name = "Ground"
floor.data.materials.append(floor_mat)

# Use a target based on the centered sequence.
target = Vector((0, 0, 0))

bpy.ops.object.camera_add(location=(2.5, -4.5, 1.4))
camera = bpy.context.object
camera.name = "MannequinCamera"
camera.rotation_euler = (target - camera.location).to_track_quat(
    "-Z", "Y"
).to_euler()
camera.data.lens = 50

scene = bpy.context.scene
scene.camera = camera

bpy.ops.object.light_add(type="AREA", location=(1.5, -3.0, 3.5))
key_light = bpy.context.object
key_light.name = "KeyLight"
key_light.data.energy = 450
key_light.data.shape = "DISK"
key_light.data.size = 4.0
key_light.rotation_euler = (
    target - key_light.location
).to_track_quat("-Z", "Y").to_euler()

bpy.ops.object.light_add(type="AREA", location=(-3.0, 1.5, 2.5))
fill_light = bpy.context.object
fill_light.name = "FillLight"
fill_light.data.energy = 250
fill_light.data.size = 3.0
fill_light.rotation_euler = (
    target - fill_light.location
).to_track_quat("-Z", "Y").to_euler()


# ============================================================
# 9. Timeline and render settings
# ============================================================

scene.render.engine = "BLENDER_EEVEE"
scene.render.fps = OUTPUT_FPS
scene.frame_start = 1
scene.frame_end = len(source_indices)

scene.render.resolution_x = 960
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUTPUT_DIR / "mannequin_frame_")

scene.world.color = (0.04, 0.04, 0.04)
scene.frame_set(1)

# Save to a NEW file. The original aist_skeleton.blend remains untouched.
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_FILE))

print("\nSUCCESS")
print("Saved new Blender project:", BLEND_FILE)
print("Frames:", len(source_indices))
print("FPS:", OUTPUT_FPS)
print("Duration:", len(source_indices) / OUTPUT_FPS, "seconds")
