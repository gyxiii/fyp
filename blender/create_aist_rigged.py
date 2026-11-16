
import bpy
import pickle
from pathlib import Path
from mathutils import Vector, Matrix

# ============================================================
# AIST++ Armature Retargeting
# Blender 5.2.2
#
# Input:
#   data/aistplusplus/keypoints3d/
#       gBR_sBM_cAll_d04_mBR0_ch01.pkl
#
# Output:
#   outputs/aistplusplus/aist_rigged.blend
#
# Existing .blend files are not opened or overwritten.
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
OUTPUT_FILE = OUTPUT_DIR / "aist_rigged.blend"

DATA_KEY = "keypoints3d_optim"

FRAME_STEP = 2
FPS = 30
UNIT_SCALE = 0.01
VOXEL_SIZE = 0.025
RENDER_SIZE = 900


# ------------------------------------------------------------
# 1. Coordinate conversion
# ------------------------------------------------------------

def convert_points(raw):
    """
    Convert source coordinates (X, Y, Z) to Blender coordinates
    (X, Z, Y), making the source Y axis vertical in Blender.
    """
    return raw[:, [0, 2, 1]].astype(float) * UNIT_SCALE


def midpoint(a, b):
    return (a + b) * 0.5


def get_points(raw, origin):
    converted = convert_points(raw)

    points = [
        Vector(v) - origin
        for v in converted
    ]

    # COCO-17 face points: nose, eyes, ears.
    head = (
        points[0]
        + points[1]
        + points[2]
        + points[3]
        + points[4]
    ) / 5.0

    shoulders = midpoint(points[5], points[6])
    hips = midpoint(points[11], points[12])

    return {
        "nose": points[0],
        "left_eye": points[1],
        "right_eye": points[2],
        "left_ear": points[3],
        "right_ear": points[4],

        "left_shoulder": points[5],
        "right_shoulder": points[6],
        "left_elbow": points[7],
        "right_elbow": points[8],
        "left_wrist": points[9],
        "right_wrist": points[10],

        "left_hip": points[11],
        "right_hip": points[12],
        "left_knee": points[13],
        "right_knee": points[14],
        "left_ankle": points[15],
        "right_ankle": points[16],

        "head": head,
        "shoulders": shoulders,
        "hips": hips,
    }


def add_head_top(points):
    """
    Estimate the top of the head from the face and shoulder positions.
    """
    up = points["head"] - points["shoulders"]

    if up.length < 1e-6:
        up = Vector((0.0, 0.0, 1.0))

    up.normalize()
    points["head_top"] = points["head"] + up * 0.18

    return points


# ------------------------------------------------------------
# 2. Skeleton definition
# ------------------------------------------------------------

# Bone names and corresponding semantic start/end landmarks.
#
# These bones are deliberately independent in this first retargeting
# version. This avoids parent-space translation errors while validating
# the basic mapping from AIST++ points to Blender pose bones.
BONES = [
    ("spine", "hips", "shoulders"),
    ("neck", "shoulders", "head"),
    ("head", "head", "head_top"),

    ("upper_arm.L", "left_shoulder", "left_elbow"),
    ("forearm.L", "left_elbow", "left_wrist"),

    ("upper_arm.R", "right_shoulder", "right_elbow"),
    ("forearm.R", "right_elbow", "right_wrist"),

    ("thigh.L", "left_hip", "left_knee"),
    ("shin.L", "left_knee", "left_ankle"),

    ("thigh.R", "right_hip", "right_knee"),
    ("shin.R", "right_knee", "right_ankle"),
]


# ------------------------------------------------------------
# 3. Materials and mesh helpers
# ------------------------------------------------------------

def make_material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)

    material.use_nodes = True

    shader = material.node_tree.nodes.get("Principled BSDF")

    if shader:
        shader.inputs["Base Color"].default_value = (
            *color,
            1.0,
        )
        shader.inputs["Roughness"].default_value = 0.75

    return material


def add_ellipsoid(name, center, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=16,
        location=center,
    )

    obj = bpy.context.object
    obj.name = name
    obj.scale = scale

    bpy.ops.object.transform_apply(
        location=False,
        rotation=False,
        scale=True,
    )

    obj.data.materials.append(material)

    for polygon in obj.data.polygons:
        polygon.use_smooth = True

    return obj


def add_segment(name, start, end, radius, material):
    direction = end - start
    center = midpoint(start, end)

    obj = add_ellipsoid(
        name,
        center,
        (
            radius,
            radius,
            max(direction.length * 0.58, radius),
        ),
        material,
    )

    if direction.length > 1e-6:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = direction.to_track_quat(
            "Z",
            "Y",
        )

        bpy.context.view_layer.objects.active = obj

        bpy.ops.object.transform_apply(
            location=False,
            rotation=True,
            scale=False,
        )

    return obj


# ------------------------------------------------------------
# 4. Create one joined, deformable mannequin mesh
# ------------------------------------------------------------

def build_body(points, material):
    parts = []

    # Head.
    parts.append(
        add_ellipsoid(
            "part_head",
            points["head"],
            (0.105, 0.105, 0.14),
            material,
        )
    )

    # Torso.
    torso_length = (
        points["shoulders"] - points["hips"]
    ).length

    parts.append(
        add_ellipsoid(
            "part_torso",
            midpoint(
                points["hips"],
                points["shoulders"],
            ),
            (
                0.115,
                0.075,
                max(torso_length * 0.56, 0.12),
            ),
            material,
        )
    )

    # Pelvis.
    parts.append(
        add_ellipsoid(
            "part_pelvis",
            points["hips"],
            (0.11, 0.075, 0.075),
            material,
        )
    )

    # Neck.
    parts.append(
        add_segment(
            "part_neck",
            points["shoulders"],
            points["head"],
            0.035,
            material,
        )
    )

    # Arms and legs.
    segment_definitions = [
        (
            "upper_arm",
            "shoulder",
            "elbow",
            0.04,
        ),
        (
            "forearm",
            "elbow",
            "wrist",
            0.03,
        ),
        (
            "thigh",
            "hip",
            "knee",
            0.055,
        ),
        (
            "shin",
            "knee",
            "ankle",
            0.038,
        ),
    ]

    for side in ("left", "right"):
        for label, start_suffix, end_suffix, radius in (
            segment_definitions
        ):
            start_key = f"{side}_{start_suffix}"
            end_key = f"{side}_{end_suffix}"

            parts.append(
                add_segment(
                    f"part_{label}_{side}",
                    points[start_key],
                    points[end_key],
                    radius,
                    material,
                )
            )

        # Hands.
        parts.append(
            add_ellipsoid(
                f"part_hand_{side}",
                points[f"{side}_wrist"],
                (0.03, 0.025, 0.04),
                material,
            )
        )

        # Feet.
        parts.append(
            add_ellipsoid(
                f"part_foot_{side}",
                points[f"{side}_ankle"]
                + Vector((0.0, 0.0, -0.025)),
                (0.035, 0.065, 0.025),
                material,
            )
        )

    # Join all parts into one mesh object.
    bpy.ops.object.select_all(action="DESELECT")

    for obj in parts:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()

    body = bpy.context.object
    body.name = "AIST_BodyMesh"
    body.data.name = "AIST_BodyMesh_Mesh"

    # Voxel remesh merges overlapping ellipsoids.
    body.data.remesh_mode = "VOXEL"
    body.data.remesh_voxel_size = VOXEL_SIZE

    bpy.ops.object.voxel_remesh()

    for polygon in body.data.polygons:
        polygon.use_smooth = True

    # Gentle smoothing.
    smooth = body.modifiers.new(
        "BodySurfaceSmooth",
        "SMOOTH",
    )
    smooth.factor = 0.7
    smooth.iterations = 3

    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(
        modifier=smooth.name
    )

    return body


# ------------------------------------------------------------
# 5. Build the Armature in the first-frame rest pose
# ------------------------------------------------------------

def build_armature(first_points):
    armature_data = bpy.data.armatures.new(
        "AIST_ArmatureData"
    )

    armature_obj = bpy.data.objects.new(
        "AIST_Armature",
        armature_data,
    )

    bpy.context.collection.objects.link(armature_obj)

    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    bpy.ops.object.mode_set(mode="EDIT")

    for name, start_key, end_key in BONES:
        start = first_points[start_key]
        end = first_points[end_key]

        # Prevent zero-length bones.
        if (end - start).length < 0.01:
            end = start + Vector((0.0, 0.0, 0.03))

        bone = armature_data.edit_bones.new(name)
        bone.head = start
        bone.tail = end

        bone.use_deform = True

    bpy.ops.object.mode_set(mode="OBJECT")

    armature_obj.show_in_front = True
    armature_data.display_type = "OCTAHEDRAL"

    return armature_obj


# ------------------------------------------------------------
# 6. Calculate vertex weights
# ------------------------------------------------------------

def distance_to_segment(point, start, end):
    direction = end - start
    length_squared = direction.length_squared

    if length_squared < 1e-12:
        return (point - start).length

    t = (point - start).dot(direction) / length_squared
    t = max(0.0, min(1.0, t))

    closest = start + direction * t

    return (point - closest).length


def assign_weights(body, armature_obj, first_points):
    # Make one vertex group per deform bone.
    groups = {}

    for name, _, _ in BONES:
        groups[name] = body.vertex_groups.new(
            name=name
        )

    segments = []

    for name, start_key, end_key in BONES:
        segments.append(
            (
                name,
                first_points[start_key].copy(),
                first_points[end_key].copy(),
            )
        )

    # Assign each vertex to its three nearest bone segments.
    # The weights are normalized to sum to 1 for each vertex.
    for vertex in body.data.vertices:
        co = vertex.co

        distances = []

        for name, start, end in segments:
            distance = distance_to_segment(
                co,
                start,
                end,
            )

            distances.append(
                (name, max(distance, 0.005))
            )

        distances.sort(key=lambda item: item[1])
        nearest = distances[:3]

        inverse_weights = [
            1.0 / (distance * distance)
            for _, distance in nearest
        ]

        total = sum(inverse_weights)

        for (name, _), weight in zip(
            nearest,
            inverse_weights,
        ):
            groups[name].add(
                [vertex.index],
                weight / total,
                "REPLACE",
            )

    modifier = body.modifiers.new(
        "AIST_ArmatureDeform",
        "ARMATURE",
    )

    modifier.object = armature_obj
    modifier.use_deform_preserve_volume = True

    # The mesh and armature share the same object-space origin.
    body.parent = armature_obj

    return body


# ------------------------------------------------------------
# 7. Retarget animation to pose bones
# ------------------------------------------------------------

def animate_armature(armature_obj, raw_frames, origin):
    scene = bpy.context.scene

    scene.frame_start = 1
    scene.frame_end = len(raw_frames)
    scene.render.fps = FPS

    # Cache rest bone data.
    rest_data = {}

    for name, _, _ in BONES:
        bone = armature_obj.data.bones[name]

        rest_data[name] = {
            "head": bone.head_local.copy(),
            "matrix": bone.matrix_local.copy(),
            "length": max(bone.length, 1e-6),
        }

    # Animate every bone using an armature-space target matrix.
    for frame_number, raw in enumerate(
        raw_frames,
        start=1,
    ):
        points = add_head_top(
            get_points(raw, origin)
        )

        scene.frame_set(frame_number)

        for name, start_key, end_key in BONES:
            start = points[start_key]
            end = points[end_key]
            direction = end - start

            if direction.length < 1e-6:
                continue

            rest = rest_data[name]
            rest_head = rest["head"]
            rest_matrix = rest["matrix"]
            rest_length = rest["length"]

            rest_bone = armature_obj.data.bones[name]

            rest_direction = (
                rest_bone.tail_local
                - rest_bone.head_local
            ).normalized()

            target_direction = direction.normalized()

            # Rotate the rest bone toward the target segment.
            delta_rotation = (
                rest_direction.rotation_difference(
                    target_direction
                )
            )

            rotation_matrix = (
                delta_rotation.to_matrix().to_4x4()
            )

            # Construct a target matrix in armature space.
            #
            # The rest head is translated to the origin,
            # the direction change is applied, then the bone
            # is placed at the target start point.
            target_matrix = (
                Matrix.Translation(start)
                @ rotation_matrix
                @ Matrix.Translation(-rest_head)
                @ rest_matrix
            )

            pose_bone = armature_obj.pose.bones[name]

            # Blender converts this armature-space matrix
            # into the bone's local pose representation.
            pose_bone.matrix = target_matrix

            # Match the target segment length.
            pose_bone.scale = (
                1.0,
                direction.length / rest_length,
                1.0,
            )

            pose_bone.keyframe_insert(
                data_path="location",
                frame=frame_number,
                group=name,
            )

            pose_bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=frame_number,
                group=name,
            )

            pose_bone.keyframe_insert(
                data_path="scale",
                frame=frame_number,
                group=name,
            )

    # Try to use linear interpolation if the current Blender
    # action exposes legacy F-Curves. Layered actions are
    # supported by Blender 5.2, so this is best-effort.
    action = (
        armature_obj.animation_data.action
        if armature_obj.animation_data
        else None
    )

    if action:
        try:
            for curve in action.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR"
        except (AttributeError, TypeError):
            pass


# ------------------------------------------------------------
# 8. Camera, lighting and ground
# ------------------------------------------------------------

def setup_scene(first_points):
    scene = bpy.context.scene

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = RENDER_SIZE
    scene.render.resolution_y = RENDER_SIZE
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.world.color = (0.025, 0.025, 0.025)

    floor_z = min(
        point.z for point in first_points.values()
    ) - 0.12

    bpy.ops.mesh.primitive_plane_add(
        size=5,
        location=(0.0, 0.0, floor_z),
    )

    floor = bpy.context.object
    floor.name = "Ground"

    floor.data.materials.append(
        make_material(
            "GroundMaterial",
            (0.07, 0.08, 0.10),
        )
    )

    # Camera.
    bpy.ops.object.camera_add(
        location=(2.6, -4.8, 1.6)
    )

    camera = bpy.context.object
    camera.name = "AIST_Camera"
    camera.data.lens = 52

    camera.rotation_euler = (
        Vector((0.0, 0.0, 0.7)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()

    scene.camera = camera

    # Key light.
    bpy.ops.object.light_add(
        type="AREA",
        location=(0.0, -3.0, 3.2),
    )

    key_light = bpy.context.object
    key_light.name = "AIST_KeyLight"
    key_light.data.energy = 450
    key_light.data.shape = "DISK"
    key_light.data.size = 4

    key_light.rotation_euler = (
        Vector((0.0, 0.0, 0.7)) - key_light.location
    ).to_track_quat("-Z", "Y").to_euler()

    # Fill light.
    bpy.ops.object.light_add(
        type="AREA",
        location=(2.5, 1.5, 2.0),
    )

    fill_light = bpy.context.object
    fill_light.name = "AIST_FillLight"
    fill_light.data.energy = 250
    fill_light.data.size = 3

    fill_light.rotation_euler = (
        Vector((0.0, 0.0, 0.7)) - fill_light.location
    ).to_track_quat("-Z", "Y").to_euler()


# ------------------------------------------------------------
# 9. Main entry point
# ------------------------------------------------------------

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"AIST++ data file not found:\n{DATA_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading AIST++ data: {DATA_FILE}")

    with open(DATA_FILE, "rb") as file:
        data = pickle.load(file)

    if DATA_KEY not in data:
        raise KeyError(
            f"Missing key '{DATA_KEY}'. "
            f"Available keys: {list(data.keys())}"
        )

    all_frames = data[DATA_KEY]

    if (
        all_frames.ndim != 3
        or all_frames.shape[1:] != (17, 3)
    ):
        raise ValueError(
            f"Expected shape (frames, 17, 3), "
            f"received {all_frames.shape}"
        )

    raw_frames = all_frames[::FRAME_STEP]

    if len(raw_frames) < 2:
        raise ValueError("Not enough frames to animate.")

    # Center the sequence using the first-frame hip midpoint.
    converted = convert_points(raw_frames[0])

    origin = Vector(
        (
            (converted[11, 0] + converted[12, 0]) / 2.0,
            (converted[11, 1] + converted[12, 1]) / 2.0,
            (converted[11, 2] + converted[12, 2]) / 2.0,
        )
    )

    first_points = add_head_top(
        get_points(raw_frames[0], origin)
    )

    # Start from a clean scene. This does not modify other .blend files.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    skin = make_material(
        "NeutralGrey",
        (0.42, 0.46, 0.50),
    )

    body = build_body(
        first_points,
        skin,
    )

    armature = build_armature(
        first_points,
    )

    assign_weights(
        body,
        armature,
        first_points,
    )

    animate_armature(
        armature,
        raw_frames,
        origin,
    )

    setup_scene(first_points)

    # Select the armature on opening.
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    bpy.context.scene.frame_set(1)

    print("Saving rigged project...")

    bpy.ops.wm.save_as_mainfile(
        filepath=str(OUTPUT_FILE)
    )

    print("=" * 60)
    print("AIST++ RIG CREATION FINISHED")
    print(f"Frames: {len(raw_frames)}")
    print(f"Bones: {len(BONES)}")
    print(f"Body vertices: {len(body.data.vertices)}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
