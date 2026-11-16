"""Render an assembled motion on a rigged 3D character (proposal section 5.8).

Input is the ``motion.npz`` written by ``src/motion_blend.py`` plus the song.
The script builds a mannequin, rigs it with the 21-bone skeleton, retargets every
frame, and renders an MP4 with the soundtrack muxed in through the VSE.

Run from a terminal (Blender in background):

    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background ^
        --factory-startup --python blender/render_character.py -- ^
        --motion outputs/runs/demo/motion.npz ^
        --audio data/audio/test_song.wav ^
        --output outputs/runs/demo/character.mp4

Useful extras:
    --blend-out outputs/runs/demo/character.blend   save the scene
    --export-fbx outputs/runs/demo/character.fbx    export a standard animation file
    --stills 3                                      render only a few stills
    --start 1 --end 120                             short test render
"""

from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from kinematics import BONES, NODE_INDEX, NODE_NAMES  # noqa: E402


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_arguments():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Render a music-driven character animation.")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blend-out", default=None)
    parser.add_argument("--export-fbx", default=None)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--camera", default="medium", choices=("wide", "medium", "close"))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stills", type=int, default=0)
    parser.add_argument("--remesh", action="store_true",
                        help="Voxel-remesh the mannequin (smoother, but weights are rebuilt from distance)")
    parser.add_argument("--no-audio", action="store_true")
    return parser.parse_args(argv)


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def make_material(name, color, roughness=0.55, metallic=0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    if shader:
        shader.inputs["Base Color"].default_value = (*color, 1.0)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
    return material


def add_ellipsoid(name, center, scale, material):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24, ring_count=16, location=tuple(center)
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def add_segment(name, start, end, radius, material):
    start, end = Vector(start), Vector(end)
    direction = end - start
    center = (start + end) * 0.5
    obj = add_ellipsoid(
        name, center,
        (radius, radius, max(direction.length * 0.58, radius)),
        material,
    )
    if direction.length > 1e-6:
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return obj


def _tag_group(obj, bone_name):
    """Give every vertex of a part a rigid weight on one bone."""
    group = obj.vertex_groups.new(name=bone_name)
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")


def build_body(nodes0: np.ndarray, material, accent=None, remesh: bool = True):
    """Build a rigid-jointed mannequin.

    Every body part is weighted 100 % to one bone.  Rigid skinning is chosen
    deliberately: the pipeline drives bones from noisy 3D joint positions, and
    rigid parts guarantee that no pose blend can stretch the mesh into spikes.
    Joint spheres hide the seams between parts.
    """
    accent = accent or material

    def node(name):
        return nodes0[NODE_INDEX[name]]

    forward = node("toe.L") - node("ankle.L")
    forward[2] = 0.0
    forward = forward / (np.linalg.norm(forward) + 1e-9)

    parts = []

    head = add_ellipsoid("part_head", node("head"), (0.105, 0.105, 0.135), material)
    _tag_group(head, "head")
    parts.append(head)

    torso_len = float(np.linalg.norm(node("chest") - node("hips")))
    torso = add_ellipsoid(
        "part_torso",
        (node("hips") + node("chest")) / 2.0,
        (0.125, 0.085, max(torso_len * 0.56, 0.12)),
        accent,
    )
    _tag_group(torso, "spine_up")
    parts.append(torso)

    pelvis = add_ellipsoid("part_pelvis", node("hips"), (0.115, 0.08, 0.08), accent)
    _tag_group(pelvis, "spine")
    parts.append(pelvis)

    neck = add_segment("part_neck", node("chest"), node("head"), 0.038, material)
    _tag_group(neck, "neck")
    parts.append(neck)

    for side in ("L", "R"):
        upper = add_segment(
            f"part_upper_arm_{side}",
            node(f"shoulder.{side}"), node(f"elbow.{side}"), 0.042, material,
        )
        _tag_group(upper, f"upper_arm.{side}")
        parts.append(upper)

        fore = add_segment(
            f"part_forearm_{side}",
            node(f"elbow.{side}"), node(f"wrist.{side}"), 0.032, material,
        )
        _tag_group(fore, f"forearm.{side}")
        parts.append(fore)

        hand = add_ellipsoid(
            f"part_hand_{side}", node(f"wrist.{side}"), (0.032, 0.026, 0.045), material
        )
        _tag_group(hand, f"hand.{side}")
        parts.append(hand)

        thigh = add_segment(
            f"part_thigh_{side}",
            node(f"hip.{side}"), node(f"knee.{side}"), 0.06, material,
        )
        _tag_group(thigh, f"thigh.{side}")
        parts.append(thigh)

        shin = add_segment(
            f"part_shin_{side}",
            node(f"knee.{side}"), node(f"ankle.{side}"), 0.042, material,
        )
        _tag_group(shin, f"shin.{side}")
        parts.append(shin)

        foot = add_ellipsoid(
            f"part_foot_{side}",
            node(f"ankle.{side}") + forward * 0.05 - np.array([0.0, 0.0, 0.03]),
            (0.042, 0.07, 0.028),
            material,
        )
        _tag_group(foot, f"foot.{side}")
        parts.append(foot)

        # Joint covers hide the seam between rigid parts.
        for bone_name, position, radius in (
            (f"upper_arm.{side}", node(f"shoulder.{side}"), 0.05),
            (f"forearm.{side}", node(f"elbow.{side}"), 0.04),
            (f"thigh.{side}", node(f"hip.{side}"), 0.07),
            (f"shin.{side}", node(f"knee.{side}"), 0.05),
        ):
            joint = add_ellipsoid(
                f"part_joint_{bone_name}_{side}", position, (radius, radius, radius), material
            )
            _tag_group(joint, bone_name)
            parts.append(joint)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()

    body = bpy.context.object
    body.name = "CharacterBody"
    body.data.name = "CharacterBodyMesh"

    if remesh:
        # Merge overlapping parts into one smooth surface, then keep the rigid
        # vertex groups that the join preserved.
        body.data.remesh_mode = "VOXEL"
        body.data.remesh_voxel_size = 0.02
        bpy.ops.object.voxel_remesh()
        for polygon in body.data.polygons:
            polygon.use_smooth = True

        smooth = body.modifiers.new("BodySmooth", "SMOOTH")
        smooth.factor = 0.5
        smooth.iterations = 2
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.modifier_apply(modifier=smooth.name)

    return body


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------

def build_armature(nodes0: np.ndarray):
    armature_data = bpy.data.armatures.new("CharacterArmatureData")
    armature = bpy.data.objects.new("CharacterArmature", armature_data)
    bpy.context.collection.objects.link(armature)

    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")

    for name, parent, child in BONES:
        head = Vector(nodes0[NODE_INDEX[parent]])
        tail = Vector(nodes0[NODE_INDEX[child]])
        if (tail - head).length < 0.015:
            tail = head + Vector((0.0, 0.0, 0.04))
        bone = armature_data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        bone.use_deform = True

    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def assign_weights_auto(body, armature, nodes0: np.ndarray):
    """Prefer Blender's heat-map weights; fall back to distance weights."""
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        print("  weights: automatic (bone heat)", flush=True)
        return True
    except RuntimeError as error:
        print(f"  automatic weights failed ({error}); using distance weights", flush=True)
        bpy.ops.object.select_all(action="DESELECT")
        body.select_set(True)
        bpy.context.view_layer.objects.active = body
        return False


def assign_weights(body, armature, nodes0: np.ndarray):
    groups = {name: body.vertex_groups.new(name=name) for name, _, _ in BONES}

    segments = []
    for name, parent, child in BONES:
        segments.append((name, Vector(nodes0[NODE_INDEX[parent]]), Vector(nodes0[NODE_INDEX[child]])))

    def distance_to_segment(point, start, end):
        direction = end - start
        length_squared = direction.length_squared
        if length_squared < 1e-12:
            return (point - start).length
        t = max(0.0, min(1.0, (point - start).dot(direction) / length_squared))
        return (point - (start + direction * t)).length

    for vertex in body.data.vertices:
        distances = [
            (name, max(distance_to_segment(vertex.co, start, end), 0.004))
            for name, start, end in segments
        ]
        distances.sort(key=lambda item: item[1])
        nearest = distances[:3]
        inverse = [1.0 / (distance * distance) for _, distance in nearest]
        total = sum(inverse)
        for (name, _), weight in zip(nearest, inverse):
            groups[name].add([vertex.index], weight / total, "REPLACE")

    modifier = body.modifiers.new("ArmatureDeform", "ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = True
    body.parent = armature


def animate(armature, nodes: np.ndarray):
    scene = bpy.context.scene
    frame_count = nodes.shape[0]
    scene.frame_start = 1
    scene.frame_end = frame_count

    rest = {}
    for name, parent, child in BONES:
        bone = armature.data.bones[name]
        rest[name] = {
            "head": bone.head_local.copy(),
            "matrix": bone.matrix_local.copy(),
            "length": max(bone.length, 1e-6),
            "direction": (bone.tail_local - bone.head_local).normalized(),
        }

    for frame_index in range(frame_count):
        frame_nodes = nodes[frame_index]
        blender_frame = frame_index + 1
        scene.frame_set(blender_frame)

        for name, parent, child in BONES:
            start = Vector(frame_nodes[NODE_INDEX[parent]])
            end = Vector(frame_nodes[NODE_INDEX[child]])
            direction = end - start
            if direction.length < 1e-6:
                continue

            data = rest[name]
            target = direction.normalized()
            delta = data["direction"].rotation_difference(target)
            target_matrix = (
                Matrix.Translation(start)
                @ delta.to_matrix().to_4x4()
                @ Matrix.Translation(-data["head"])
                @ data["matrix"]
            )

            pose_bone = armature.pose.bones[name]
            pose_bone.matrix = target_matrix
            pose_bone.keyframe_insert("location", frame=blender_frame, group=name)
            pose_bone.keyframe_insert("rotation_quaternion", frame=blender_frame, group=name)

        if blender_frame % 60 == 0:
            print(f"  keyframed {blender_frame}/{frame_count}", flush=True)

    action = armature.animation_data.action if armature.animation_data else None
    if action:
        try:
            for curve in action.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR"
        except (AttributeError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def _pick_engine(scene) -> str:
    """Choose an available EEVEE identifier across Blender versions."""
    try:
        items = [
            item.identifier
            for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        ]
    except Exception:
        items = []

    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if candidate in items:
            return candidate
    return "BLENDER_EEVEE"


def setup_world_and_lights(scene, nodes0):
    scene.render.engine = _pick_engine(scene)

    # Preview-speed EEVEE settings: the project renders hundreds of frames on a
    # laptop, so 16 TAA samples is the quality/throughput compromise.
    try:
        scene.eevee.taa_render_samples = 16
        scene.eevee.taa_samples = 8
    except AttributeError:
        pass

    if scene.world is None:
        scene.world = bpy.data.worlds.new("CharacterWorld")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.02, 0.022, 0.03, 1.0)
        background.inputs["Strength"].default_value = 0.35

    floor_z = float(np.min(nodes0[:, 2])) - 0.01
    bpy.ops.mesh.primitive_plane_add(size=14, location=(0.0, 0.0, floor_z))
    floor = bpy.context.object
    floor.name = "StageFloor"
    floor.data.materials.append(make_material("StageMaterial", (0.035, 0.038, 0.05), 0.8))

    bpy.ops.object.light_add(type="AREA", location=(1.8, -3.2, 3.4))
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 330
    key.data.shape = "DISK"
    key.data.size = 3.0
    key.rotation_euler = (Vector((0, 0, 0.9)) - key.location).to_track_quat("-Z", "Y").to_euler()

    bpy.ops.object.light_add(type="AREA", location=(-2.6, 1.6, 2.2))
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 130
    fill.data.size = 3.0
    fill.rotation_euler = (Vector((0, 0, 0.9)) - fill.location).to_track_quat("-Z", "Y").to_euler()

    bpy.ops.object.light_add(type="AREA", location=(0.0, 2.8, 2.0))
    rim = bpy.context.object
    rim.name = "RimLight"
    rim.data.energy = 200
    rim.data.size = 2.5
    rim.rotation_euler = (Vector((0, 0, 1.1)) - rim.location).to_track_quat("-Z", "Y").to_euler()


def setup_camera(scene, nodes, preset: str):
    presets = {
        "wide": (3.2, -5.6, 2.2),
        "medium": (2.2, -3.9, 1.65),
        "close": (1.8, -3.2, 1.6),
    }
    location = presets.get(preset, presets["medium"])

    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "CharacterCamera"
    camera.data.lens = 50

    target_height = 0.95
    camera.rotation_euler = (
        Vector((0.0, 0.0, target_height)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera


def configure_video(scene, output_path: Path):
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.ffmpeg.audio_channels = "STEREO"
    scene.render.ffmpeg.audio_mixrate = 44100
    scene.render.filepath = str(output_path)
    scene.render.use_file_extension = True


def create_mix_scene(animation_scene, audio_path, duration, resolution, fps):
    mix = bpy.data.scenes.new("CharacterMix")
    mix.render.engine = animation_scene.render.engine
    mix.render.resolution_x, mix.render.resolution_y = resolution
    mix.render.resolution_percentage = 100
    mix.render.fps = fps
    mix.render.fps_base = 1.0
    mix.frame_start = 1
    mix.frame_end = max(2, math.ceil(duration * fps) + 1)
    mix.render.use_sequencer = True

    editor = mix.sequence_editor_create()
    editor.strips.new_scene(name="AnimationVideo", scene=animation_scene, channel=1, frame_start=1)
    if audio_path is not None:
        strip = editor.strips.new_sound(
            name="MusicAudio", filepath=str(audio_path), channel=2, frame_start=1
        )
        strip.volume = 1.0
    return mix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    motion_path = Path(args.motion).resolve()
    output_path = Path(args.output).resolve()
    audio_path = Path(args.audio).resolve() if args.audio else None

    if not motion_path.exists():
        raise FileNotFoundError(f"motion.npz not found: {motion_path}")
    if audio_path is not None and not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    data = np.load(motion_path)
    nodes = data["nodes"].astype(np.float64)
    fps = int(data["fps"])

    start = max(0, int(args.start) - 1) if args.start else 0
    end = int(args.end) if args.end else nodes.shape[0]
    nodes = nodes[start:end]

    width, height = (int(value) for value in args.resolution.lower().split("x"))

    print(f"Motion   : {motion_path} ({nodes.shape[0]} frames @ {fps} FPS)", flush=True)
    print(f"Output   : {output_path}", flush=True)

    # Clean factory scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_sequencer = False

    skin = make_material("CharacterSkin", (0.46, 0.50, 0.56), roughness=0.5)
    accent = make_material("CharacterAccent", (0.10, 0.35, 0.62), roughness=0.45)

    body = build_body(nodes[0], skin, accent, remesh=bool(args.remesh))
    armature = build_armature(nodes[0])

    modifier = body.modifiers.new("ArmatureDeform", "ARMATURE")
    modifier.object = armature
    modifier.use_deform_preserve_volume = False
    body.parent = armature

    animate(armature, nodes)

    setup_world_and_lights(scene, nodes[0])
    setup_camera(scene, nodes, args.camera)

    bpy.context.scene.frame_set(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stills:
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
        picks = np.linspace(1, max(1, nodes.shape[0]), args.stills).astype(int)
        for index, frame in enumerate(picks, start=1):
            scene.frame_set(int(frame))
            scene.render.filepath = str(output_path.parent / f"still_{index:02d}.png")
            bpy.ops.render.render(write_still=True)
            print(f"still {index}: frame {frame}", flush=True)
        return

    if args.blend_out:
        blend_path = Path(args.blend_out).resolve()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"Saved blend: {blend_path}", flush=True)

    if args.export_fbx:
        fbx_path = Path(args.export_fbx).resolve()
        fbx_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        body.select_set(True)
        try:
            bpy.ops.export_scene.fbx(
                filepath=str(fbx_path),
                use_selection=True,
                bake_anim=True,
                add_leaf_bones=False,
                object_scale=1.0,
            )
            print(f"Exported FBX: {fbx_path}", flush=True)
        except Exception as error:  # FBX add-on may be disabled
            print(f"FBX export skipped: {error}", flush=True)

    duration = nodes.shape[0] / fps
    if args.no_audio or audio_path is None:
        configure_video(scene, output_path)
        bpy.ops.render.render(animation=True)
    else:
        duration = min(duration, audio_duration(audio_path))
        mix = create_mix_scene(scene, audio_path, duration, (width, height), fps)
        configure_video(mix, output_path)
        bpy.ops.render.render(animation=True, scene=mix.name)

    expected = output_path
    if not expected.exists():
        alternative = Path(str(output_path) + ".mp4")
        if alternative.exists():
            expected = alternative

    if not expected.exists():
        raise RuntimeError(f"Render finished but no MP4 was produced: {output_path}")

    print("RENDER_SUCCESS:", expected, flush=True)
    print(f"Size: {expected.stat().st_size / (1024 * 1024):.2f} MB", flush=True)


if __name__ == "__main__":
    main()
