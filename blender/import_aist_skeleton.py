
import bpy
import pickle
import math
from pathlib import Path

from mathutils import Vector
import numpy as np


# ============================================================
# 1. Paths and settings
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

BLEND_FILE = OUTPUT_DIR / "aist_skeleton.blend"

SOURCE_FPS = 60
FRAME_STEP = 2
OUTPUT_FPS = 30

DATA_KEY = "keypoints3d_optim"

# Correct COCO-17 connections.
EDGES = [
    # Face
    (0, 1), (0, 2),
    (1, 3), (2, 4),

    # Shoulders and torso
    (5, 6),
    (5, 11), (6, 12), (11, 12),

    # Arms
    (5, 7), (7, 9),
    (6, 8), (8, 10),

    # Legs
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Convert the dataset's coordinate scale to a manageable
# Blender scene size. The displayed axes will be:
# Blender X = source X
# Blender Y = source Z (depth)
# Blender Z = source Y (vertical)
UNIT_SCALE = 0.01

JOINT_RADIUS = 0.035
BONE_RADIUS = 0.018


# ============================================================
# 2. Load keypoint data
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

with open(DATA_FILE, "rb") as f:
    data = pickle.load(f, encoding="latin1")

if DATA_KEY not in data:
    raise KeyError(
        f"Missing {DATA_KEY}. Available keys: {list(data.keys())}"
    )

keypoints = np.asarray(data[DATA_KEY], dtype=np.float32)

if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
    raise ValueError(f"Unexpected keypoint shape: {keypoints.shape}")

if not np.isfinite(keypoints).all():
    raise ValueError("The keypoint data contains NaN or infinity.")

source_frame_count = keypoints.shape[0]
frame_indices = list(range(0, source_frame_count, FRAME_STEP))
frame_count = len(frame_indices)

print("Loaded keypoints:", keypoints.shape)
print("Animation frames:", frame_count)


# ============================================================
# 3. Convert coordinates and center the motion
# ============================================================

# Swap Y and Z so the original Y axis becomes Blender's Z-up.
points = keypoints[:, :, [0, 2, 1]].copy()

minimum = points.reshape(-1, 3).min(axis=0)
maximum = points.reshape(-1, 3).max(axis=0)
center = (minimum + maximum) / 2.0

points = (points - center) * UNIT_SCALE


# ============================================================
# 4. Clear the default scene
# ============================================================

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

for datablocks in (
    bpy.data.meshes,
    bpy.data.curves,
    bpy.data.cameras,
    bpy.data.lights,
):
    # Remove unused data blocks left from the default scene.
    for block in list(datablocks):
        if block.users == 0:
            datablocks.remove(block)


# ============================================================
# 5. Materials
# ============================================================

def make_material(name, color):
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True

    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.45

    return material


joint_material = make_material(
    "Joint Blue",
    (0.08, 0.38, 0.95),
)

bone_material = make_material(
    "Bone Light Blue",
    (0.15, 0.75, 0.95),
)

floor_material = make_material(
    "Floor",
    (0.045, 0.055, 0.075),
)


# ============================================================
# 6. Create joint spheres
# ============================================================

joints = []

for joint_index in range(17):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=12,
        ring_count=8,
        radius=JOINT_RADIUS,
        location=(0, 0, 0),
    )

    obj = bpy.context.object
    obj.name = f"Joint_{joint_index:02d}"
    obj.data.materials.append(joint_material)
    bpy.ops.object.shade_smooth()

    joints.append(obj)


# ============================================================
# 7. Create bone cylinders
# ============================================================

bones = []

for edge_index, (start_joint, end_joint) in enumerate(EDGES):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=10,
        radius=BONE_RADIUS,
        depth=1.0,
        location=(0, 0, 0),
    )

    obj = bpy.context.object
    obj.name = f"Bone_{start_joint:02d}_{end_joint:02d}"
    obj.data.materials.append(bone_material)
    bpy.ops.object.shade_smooth()

    bones.append((obj, start_joint, end_joint))


# ============================================================
# 8. Animate every joint and bone
# ============================================================

scene = bpy.context.scene
scene.render.fps = OUTPUT_FPS
scene.frame_start = 1
scene.frame_end = frame_count

for output_index, source_index in enumerate(frame_indices):
    blender_frame = output_index + 1
    frame_points = points[source_index]

    # Animate joint centers.
    for joint_index, obj in enumerate(joints):
        p = Vector(frame_points[joint_index])
        obj.location = p
        obj.keyframe_insert(
            data_path="location",
            frame=blender_frame,
        )

    # Animate the cylinders between the joint pairs.
    for obj, start_joint, end_joint in bones:
        start = Vector(frame_points[start_joint])
        end = Vector(frame_points[end_joint])

        direction = end - start
        length = direction.length

        midpoint = (start + end) / 2.0

        obj.location = midpoint
        obj.keyframe_insert(
            data_path="location",
            frame=blender_frame,
        )

        if length > 1e-6:
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
            obj.scale = (1.0, 1.0, length)
        else:
            obj.scale = (1.0, 1.0, 0.0001)

        obj.keyframe_insert(
            data_path="rotation_quaternion",
            frame=blender_frame,
        )
        obj.keyframe_insert(
            data_path="scale",
            frame=blender_frame,
        )

    if blender_frame % 60 == 1:
        print(f"Processed frame {blender_frame}/{frame_count}")


# Make motion follow the sampled source poses without overshooting.
for obj in joints + [item[0] for item in bones]:
    animation_data = obj.animation_data

    if not animation_data or not animation_data.action:
        continue

    action = animation_data.action

    # Blender 5.2 may use layered Actions.
    # Try the legacy API first, then the layered API.
    if hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            for key in fcurve.keyframe_points:
                key.interpolation = "LINEAR"
    else:
        for layer in action.layers:
            for strip in layer.strips:
                if strip.type != "KEYFRAME":
                    continue

                for slot in action.slots:
                    channelbag = strip.channelbag(slot)

                    if channelbag is None:
                        continue

                    for fcurve in channelbag.fcurves:
                        for key in fcurve.keyframe_points:
                            key.interpolation = "LINEAR"

# ============================================================
# 9. Ground, camera, and lights
# ============================================================

bpy.ops.mesh.primitive_plane_add(
    size=8,
    location=(0, 0, -1.0),
)

floor = bpy.context.object
floor.name = "Ground"
floor.data.materials.append(floor_material)

# Keep the ground stationary and outside the character animation.
floor.select_set(False)

bpy.ops.object.camera_add(
    location=(2.7, -4.5, 1.7),
)

camera = bpy.context.object
camera.name = "SkeletonCamera"

target = Vector((0, 0, 0))
camera.rotation_euler = (target - camera.location).to_track_quat(
    "-Z", "Y"
).to_euler()

camera.data.lens = 50
scene.camera = camera

bpy.ops.object.light_add(
    type="AREA",
    location=(1.5, -3.0, 4.0),
)

key_light = bpy.context.object
key_light.name = "KeyLight"
key_light.data.energy = 450
key_light.data.shape = "DISK"
key_light.data.size = 4.0

key_light.rotation_euler = (
    Vector((0, 0, 0)) - key_light.location
).to_track_quat("-Z", "Y").to_euler()

bpy.ops.object.light_add(
    type="AREA",
    location=(-3.0, 1.5, 2.5),
)

fill_light = bpy.context.object
fill_light.name = "FillLight"
fill_light.data.energy = 250
fill_light.data.size = 3.0

fill_light.rotation_euler = (
    Vector((0, 0, 0)) - fill_light.location
).to_track_quat("-Z", "Y").to_euler()


# ============================================================
# 10. Render and timeline settings
# ============================================================

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 960
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUTPUT_DIR / "aist_frame_")

scene.world.color = (0.025, 0.025, 0.025)

scene.frame_set(1)

# Select the first joint for a convenient starting state.
bpy.ops.object.select_all(action="DESELECT")
joints[0].select_set(True)
bpy.context.view_layer.objects.active = joints[0]

# Save the native Blender project.
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_FILE))

print("\nSUCCESS")
print("Blender project:", BLEND_FILE)
print("Frames:", frame_count)
print("FPS:", OUTPUT_FPS)
print("Duration:", frame_count / OUTPUT_FPS, "seconds")
print("Open this .blend file in Blender and press Space to play.")
