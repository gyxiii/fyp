
import bpy
import math
from pathlib import Path
from mathutils import Vector

# ============================================================
# AIST++ Skin Repair
# Keeps the existing armature and its animation.
# Rebuilds the mesh using rigidly weighted body segments.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "aistplusplus"
    / "aist_rigged.blend"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "aistplusplus"
    / "aist_rigged_rigid_skin.blend"
)

SPHERE_SEGMENTS = 16
SPHERE_RINGS = 12

# Radius of each segment in Blender scene units.
RADII = {
    "spine": 0.105,
    "neck": 0.045,
    "head": 0.105,
    "upper_arm": 0.045,
    "forearm": 0.035,
    "thigh": 0.065,
    "shin": 0.043,
}

# ------------------------------------------------------------
# 1. Open the existing project
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find the existing Blender file:\n{INPUT_FILE}"
    )

bpy.ops.wm.open_mainfile(filepath=str(INPUT_FILE))

scene = bpy.context.scene

armatures = [
    obj for obj in scene.objects
    if obj.type == "ARMATURE"
]

if not armatures:
    raise RuntimeError(
        "No armature was found in aist_rigged.blend."
    )

# Prefer the armature that owns the existing animated bones.
armature = next(
    (
        obj for obj in armatures
        if obj.animation_data is not None
        and obj.animation_data.action is not None
    ),
    armatures[0],
)

print("Using armature:", armature.name)
print("Existing action:", (
    armature.animation_data.action.name
    if armature.animation_data
    and armature.animation_data.action
    else "No direct action found"
))

# Use the armature's actual rest-bone positions.
# This does not recalculate or overwrite the animation.
bone_names = set(armature.data.bones.keys())

required_bones = [
    "spine",
    "neck",
    "head",
    "upper_arm.L",
    "forearm.L",
    "upper_arm.R",
    "forearm.R",
    "thigh.L",
    "shin.L",
    "thigh.R",
    "shin.R",
]

missing = [
    name for name in required_bones
    if name not in bone_names
]

if missing:
    raise RuntimeError(
        "Expected bones were not found: "
        + ", ".join(missing)
        + "\nAvailable bones: "
        + ", ".join(sorted(bone_names))
    )

# ------------------------------------------------------------
# 2. Remove old armature-deformed mesh objects only
# ------------------------------------------------------------

old_meshes = []

for obj in list(scene.objects):
    if obj.type != "MESH":
        continue

    has_armature_modifier = any(
        mod.type == "ARMATURE"
        and mod.object == armature
        for mod in obj.modifiers
    )

    if has_armature_modifier:
        old_meshes.append(obj)

for obj in old_meshes:
    bpy.data.objects.remove(obj, do_unlink=True)

print("Removed old deformed meshes:", len(old_meshes))

# ------------------------------------------------------------
# 3. Segment dimensions
# ------------------------------------------------------------

def get_radius(bone_name):
    if bone_name.startswith("upper_arm"):
        return RADII["upper_arm"]

    if bone_name.startswith("forearm"):
        return RADII["forearm"]

    if bone_name.startswith("thigh"):
        return RADII["thigh"]

    if bone_name.startswith("shin"):
        return RADII["shin"]

    if bone_name == "neck":
        return RADII["neck"]

    if bone_name == "head":
        return RADII["head"]

    return RADII["spine"]


def get_segment_length(bone):
    return (bone.tail_local - bone.head_local).length


# ------------------------------------------------------------
# 4. Create one ellipsoid for each bone
#    Every vertex belongs to exactly one bone group.
# ------------------------------------------------------------

created_parts = []

for bone_name in required_bones:
    bone = armature.data.bones[bone_name]

    # Bone rest positions are in armature-local coordinates.
    # Convert them to world coordinates for mesh construction.
    start = armature.matrix_world @ bone.head_local
    end = armature.matrix_world @ bone.tail_local

    direction = end - start
    length = direction.length

    if length < 1e-6:
        print("Skipping zero-length bone:", bone_name)
        continue

    midpoint = (start + end) * 0.5
    radius = get_radius(bone_name)

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=SPHERE_SEGMENTS,
        ring_count=SPHERE_RINGS,
        radius=1.0,
        location=midpoint,
    )

    part = bpy.context.object
    part.name = "SkinPart_" + bone_name

    # A sphere's local Z axis is aligned with the bone.
    part.rotation_mode = "QUATERNION"
    part.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(
        direction.normalized()
    )

    # Stretch along the bone direction, not across the body.
    # The overlapping ends reduce gaps at joints.
    part.scale = (
        radius,
        radius,
        length * 0.5 + radius * 0.35,
    )

    # Apply geometry transforms before joining.
    bpy.ops.object.transform_apply(
        location=False,
        rotation=True,
        scale=True,
    )

    # Give all vertices of this segment 100% influence
    # from the corresponding bone.
    group = part.vertex_groups.new(name=bone_name)

    if len(part.data.vertices) > 0:
        group.add(
            list(range(len(part.data.vertices))),
            1.0,
            "REPLACE",
        )

    created_parts.append(part)

# ------------------------------------------------------------
# 5. Join the segments into one skin mesh
# ------------------------------------------------------------

if not created_parts:
    raise RuntimeError("No skin segments were created.")

bpy.ops.object.select_all(action="DESELECT")

for part in created_parts:
    part.select_set(True)

bpy.context.view_layer.objects.active = created_parts[0]
bpy.ops.object.join()

skin = bpy.context.object
skin.name = "AIST_RigidSkin"

# Ensure the mesh is not parented to another object that would
# introduce an unintended additional transform.
skin.parent = None
skin.matrix_world.identity()

# ------------------------------------------------------------
# 6. Bind the new mesh to the EXISTING armature
# ------------------------------------------------------------

# Remove any old armature modifiers on the new object.
for mod in list(skin.modifiers):
    if mod.type == "ARMATURE":
        skin.modifiers.remove(mod)

modifier = skin.modifiers.new(
    name="AIST_Armature",
    type="ARMATURE",
)

modifier.object = armature
modifier.use_vertex_groups = True
modifier.use_bone_envelopes = False

# No automatic weights are generated here.
# The explicit vertex groups above are retained.

# ------------------------------------------------------------
# 7. Preserve timeline and save to a new file
# ------------------------------------------------------------

scene.frame_set(scene.frame_start)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_FILE))

print("\n" + "=" * 55)
print("AIST++ SKIN REPAIR FINISHED")
print("=" * 55)
print("Armature:", armature.name)
print("Skin mesh:", skin.name)
print("Skin vertices:", len(skin.data.vertices))
print("Skin segments:", len(created_parts))
print("Original animation was not intentionally modified.")
print("Saved to:", OUTPUT_FILE)
