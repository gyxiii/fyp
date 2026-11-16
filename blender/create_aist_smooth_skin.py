
import bpy
import math
from pathlib import Path
from mathutils import Vector

# ============================================================
# AIST++ Continuous Skin Builder
# Input:  aist_rigged_rigid_skin.blend
# Output: aist_rigged_smooth.blend
#
# Keeps the existing armature and animation.
# Rebuilds the skin as one voxel-remeshed surface.
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "outputs" / "aistplusplus"
    / "aist_rigged_rigid_skin.blend"
)

OUTPUT_FILE = (
    PROJECT_DIR / "outputs" / "aistplusplus"
    / "aist_rigged_smooth.blend"
)

VOXEL_SIZE = 0.012
SMOOTH_FACTOR = 0.35
SMOOTH_STEPS = 3

SPHERE_SEGMENTS = 20
SPHERE_RINGS = 12

# Approximate body radii in Blender scene units.
RADII = {
    "spine": 0.115,
    "neck": 0.050,
    "head": 0.105,
    "upper_arm": 0.047,
    "forearm": 0.037,
    "thigh": 0.065,
    "shin": 0.045,
}

BONE_NAMES = [
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


def log(message):
    print("[AIST Smooth Skin]", message)


def bone_radius(name):
    if name.startswith("upper_arm"):
        return RADII["upper_arm"]
    if name.startswith("forearm"):
        return RADII["forearm"]
    if name.startswith("thigh"):
        return RADII["thigh"]
    if name.startswith("shin"):
        return RADII["shin"]
    if name == "neck":
        return RADII["neck"]
    if name == "head":
        return RADII["head"]
    return RADII["spine"]


def point_segment_distance(point, start, end):
    segment = end - start
    length_squared = segment.length_squared

    if length_squared < 1e-12:
        return (point - start).length

    t = (point - start).dot(segment) / length_squared
    t = max(0.0, min(1.0, t))
    closest = start + t * segment

    return (point - closest).length


# ------------------------------------------------------------
# 1. Load the existing animation file
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

bpy.ops.wm.open_mainfile(filepath=str(INPUT_FILE))

scene = bpy.context.scene

armatures = [
    obj for obj in scene.objects
    if obj.type == "ARMATURE"
]

if not armatures:
    raise RuntimeError("No armature found in the input file.")

# Prefer the armature with the existing animation action.
armature = next(
    (
        obj for obj in armatures
        if obj.animation_data is not None
        and obj.animation_data.action is not None
    ),
    armatures[0],
)

log(f"Armature: {armature.name}")

if not armature.data.bones:
    raise RuntimeError("The armature has no bones.")

missing = [
    name for name in BONE_NAMES
    if name not in armature.data.bones
]

if missing:
    raise RuntimeError(
        "Required bones are missing: " + ", ".join(missing)
    )

# ------------------------------------------------------------
# 2. Capture bone REST positions.
#    Never use the current animated pose for mesh construction.
# ------------------------------------------------------------

bpy.context.view_layer.update()

segments = []

for name in BONE_NAMES:
    bone = armature.data.bones[name]

    start = armature.matrix_world @ bone.head_local
    end = armature.matrix_world @ bone.tail_local

    if (end - start).length < 1e-5:
        log(f"Skipping zero-length bone: {name}")
        continue

    segments.append({
        "name": name,
        "start": start.copy(),
        "end": end.copy(),
        "radius": bone_radius(name),
    })

if not segments:
    raise RuntimeError("No valid bone segments were found.")

# Additional round masses at the wrists and ankles help
# the remeshed surface bridge nearby segment ends.
joint_points = []

for name in [
    "forearm.L",
    "forearm.R",
    "shin.L",
    "shin.R",
]:
    bone = armature.data.bones[name]
    joint_points.append(
        (
            armature.matrix_world @ bone.tail_local,
            bone_radius(name) * 0.90,
        )
    )

# ------------------------------------------------------------
# 3. Remove only the old skin meshes.
#    Keep the armature, actions, camera and lights.
# ------------------------------------------------------------

for obj in list(scene.objects):
    if obj.type != "MESH":
        continue

    is_old_skin = (
        obj.name == "AIST_RigidSkin"
        or obj.name.startswith("SkinPart_")
        or any(
            mod.type == "ARMATURE"
            and mod.object == armature
            for mod in obj.modifiers
        )
    )

    if is_old_skin:
        bpy.data.objects.remove(obj, do_unlink=True)

log("Removed old skin meshes.")

# ------------------------------------------------------------
# 4. Build overlapping ellipsoids along REST bone segments.
#    They will be fused into one surface in the next step.
# ------------------------------------------------------------

parts = []

def add_ellipsoid(location, radius_x, radius_y, radius_z,
                  direction=None, name="SkinTemp"):

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=SPHERE_SEGMENTS,
        ring_count=SPHERE_RINGS,
        radius=1.0,
        location=location,
    )

    obj = bpy.context.object
    obj.name = name

    if direction is not None and direction.length > 1e-6:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Vector(
            (0.0, 0.0, 1.0)
        ).rotation_difference(direction.normalized())

    obj.scale = (radius_x, radius_y, radius_z)

    bpy.ops.object.transform_apply(
        location=False,
        rotation=True,
        scale=True,
    )

    parts.append(obj)
    return obj


for segment in segments:
    start = segment["start"]
    end = segment["end"]
    direction = end - start
    length = direction.length
    radius = segment["radius"]

    # Slightly overlap adjacent parts so voxel remeshing can
    # fuse their surfaces.
    add_ellipsoid(
        location=(start + end) * 0.5,
        radius_x=radius,
        radius_y=radius,
        radius_z=length * 0.5 + radius * 0.65,
        direction=direction,
        name="Temp_" + segment["name"],
    )

for index, (point, radius) in enumerate(joint_points):
    add_ellipsoid(
        location=point,
        radius_x=radius,
        radius_y=radius,
        radius_z=radius,
        name=f"Temp_Joint_{index}",
    )

if not parts:
    raise RuntimeError("No mesh parts were generated.")

# ------------------------------------------------------------
# 5. Join temporary parts, then voxel-remesh into one surface.
# ------------------------------------------------------------

bpy.ops.object.select_all(action="DESELECT")

for obj in parts:
    obj.select_set(True)

bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()

skin = bpy.context.object
skin.name = "AIST_SmoothSkin"

# Object coordinates are kept in world-space alignment with
# the rest skeleton used above.
skin.parent = None
skin.matrix_world.identity()

# Blender's voxel remesh merges intersecting surfaces.
skin.data.remesh_voxel_size = VOXEL_SIZE
skin.data.remesh_mode = "VOXEL"

bpy.context.view_layer.objects.active = skin
skin.select_set(True)

bpy.ops.object.voxel_remesh()

log(f"Voxel remesh complete: {len(skin.data.vertices)} vertices.")

if len(skin.data.vertices) == 0:
    raise RuntimeError("Voxel remesh produced an empty mesh.")

# Smooth voxel artifacts without adding another deformation
# modifier that could affect animation.
smooth = skin.modifiers.new(
    name="TemporarySurfaceSmooth",
    type="SMOOTH",
)

smooth.factor = SMOOTH_FACTOR
smooth.iterations = SMOOTH_STEPS

bpy.context.view_layer.objects.active = skin
bpy.ops.object.modifier_apply(modifier=smooth.name)

# Recalculate normals.
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")

for polygon in skin.data.polygons:
    polygon.use_smooth = True

# ------------------------------------------------------------
# 6. Create vertex groups with localized, smooth weights.
#    Each vertex influences only nearby bones, not every bone.
# ------------------------------------------------------------

groups = {}

for segment in segments:
    groups[segment["name"]] = skin.vertex_groups.new(
        name=segment["name"]
    )

# Use inverse-distance weighting for the nearest local segments.
# A distance cutoff prevents faraway bones from pulling vertices.
MAX_INFLUENCES = 3

for vertex in skin.data.vertices:
    point = skin.matrix_world @ vertex.co

    distances = []

    for segment in segments:
        distance = point_segment_distance(
            point,
            segment["start"],
            segment["end"],
        )

        distances.append((distance, segment))

    distances.sort(key=lambda item: item[0])

    selected = []

    for distance, segment in distances:
        # Keep only nearby bones. The tolerance is scaled to
        # the radius of each segment.
        if distance <= segment["radius"] * 2.5:
            selected.append((distance, segment))

        if len(selected) >= MAX_INFLUENCES:
            break

    # Fallback: the closest segment always receives influence.
    if not selected:
        selected = [distances[0]]

    raw_weights = []

    for distance, segment in selected:
        radius = max(segment["radius"], 1e-4)

        # Smooth, local weighting. Closer bones receive more weight.
        weight = math.exp(
            -2.0 * (distance / radius) ** 2
        )

        raw_weights.append(weight)

    total = sum(raw_weights)

    if total < 1e-12:
        raw_weights = [1.0] + [0.0] * (len(selected) - 1)
        total = 1.0

    for (distance, segment), weight in zip(
        selected, raw_weights
    ):
        if weight <= 1e-8:
            continue

        groups[segment["name"]].add(
            [vertex.index],
            weight / total,
            "REPLACE",
        )

log("Localized vertex weights assigned.")

# ------------------------------------------------------------
# 7. Bind the continuous mesh to the existing armature.
# ------------------------------------------------------------

for modifier in list(skin.modifiers):
    if modifier.type == "ARMATURE":
        skin.modifiers.remove(modifier)

armature_modifier = skin.modifiers.new(
    name="AIST_Armature",
    type="ARMATURE",
)

armature_modifier.object = armature
armature_modifier.use_vertex_groups = True
armature_modifier.use_bone_envelopes = False

# Do not parent the mesh to the armature object as well:
# the Armature modifier alone drives the deformation.
skin.parent = None
skin.matrix_world.identity()

# Preserve the original animation range and start frame.
scene.frame_set(scene.frame_start)

# ------------------------------------------------------------
# 8. Save as a NEW file.
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.save_as_mainfile(
    filepath=str(OUTPUT_FILE)
)

log("=" * 50)
log("CONTINUOUS SKIN BUILD FINISHED")
log(f"Armature: {armature.name}")
log(f"Skin: {skin.name}")
log(f"Vertices: {len(skin.data.vertices)}")
log(f"Bones weighted: {len(groups)}")
log(f"Saved: {OUTPUT_FILE}")
log("Original test file was not overwritten.")
