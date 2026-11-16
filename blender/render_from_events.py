
import argparse
import math
import sys
import wave
from pathlib import Path

import bpy
from mathutils import Vector


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    argv = sys.argv

    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render music-driven 3D animation to MP4."
    )

    parser.add_argument("--events", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--intensity", type=float, default=1.0)

    return parser.parse_args(argv)


# ============================================================
# Utilities
# ============================================================

def get_audio_duration(audio_path):
    """Read the duration of a WAV file."""
    with wave.open(str(audio_path), "rb") as audio:
        frames = audio.getnframes()
        sample_rate = audio.getframerate()

        if sample_rate <= 0:
            raise ValueError("Invalid WAV sample rate.")

        return frames / sample_rate


def load_events(events_path):
    """Read animation events from a CSV file."""
    import csv

    events = []

    with open(events_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"time_sec", "event_type", "intensity"}

        if not reader.fieldnames:
            raise ValueError("The events CSV has no header.")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                "Missing CSV columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            try:
                time_sec = float(row["time_sec"])
                intensity = float(row["intensity"])
                event_type = str(row["event_type"])

                if not math.isfinite(time_sec):
                    continue

                if not math.isfinite(intensity):
                    continue

                if time_sec < 0:
                    continue

                events.append({
                    "time_sec": time_sec,
                    "event_type": event_type,
                    "intensity": max(0.0, intensity),
                })

            except (TypeError, ValueError):
                continue

    events.sort(key=lambda item: item["time_sec"])

    if not events:
        raise ValueError("No valid animation events were found.")

    return events


def make_material(name, color, metallic=0.0, roughness=0.4):
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = (*color, 1.0)
    material.use_nodes = True

    bsdf = material.node_tree.nodes.get("Principled BSDF")

    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness

    return material


# ============================================================
# Scene setup
# ============================================================

def clear_scene(scene):
    """Remove objects from the current factory-startup scene."""
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def create_animation_scene(scene, duration, events, speed, intensity_scale):
    fps = 30

    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = fps
    scene.render.fps_base = 1.0

    scene.frame_start = 1
    scene.frame_end = max(2, math.ceil(duration * fps) + 1)

    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGB"
    scene.render.use_sequencer = False

    # World background
    if scene.world is None:
        scene.world = bpy.data.worlds.new("AnimationWorld")

    scene.world.color = (0.025, 0.025, 0.04)

    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")

    if background:
        background.inputs["Color"].default_value = (
            0.025, 0.025, 0.04, 1.0
        )
        background.inputs["Strength"].default_value = 0.5

    # Materials
    cube_material = make_material(
        "CubeMaterial",
        (0.08, 0.42, 0.95),
        metallic=0.2,
        roughness=0.28,
    )

    ground_material = make_material(
        "GroundMaterial",
        (0.07, 0.085, 0.12),
        metallic=0.05,
        roughness=0.65,
    )

    # Ground
    bpy.ops.mesh.primitive_plane_add(
        size=200,
        location=(0, 0, 0),
    )

    ground = bpy.context.object
    ground.name = "Ground"
    ground.data.materials.append(ground_material)

    # Animated cube
    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, 0.5),
    )

    cube = bpy.context.object
    cube.name = "MusicDrivenCube"
    cube.data.materials.append(cube_material)

    bevel = cube.modifiers.new("SoftEdges", "BEVEL")
    bevel.width = 0.08
    bevel.segments = 4

    normals = cube.modifiers.new("WeightedNormals", "WEIGHTED_NORMAL")

    # Starting pose
    base_z = 0.5
    base_scale = 1.0

    cube.location.z = base_z
    cube.scale = (base_scale, base_scale, base_scale)

    cube.keyframe_insert(data_path="location", frame=1)
    cube.keyframe_insert(data_path="scale", frame=1)

    # Convert musical events into animation keyframes.
    # speed > 1 makes the bounce action faster.
    safe_speed = max(0.1, speed)
    bounce_frames = max(2, round(8 / safe_speed))
    settle_frames = max(3, round(13 / safe_speed))

    for event in events:
        event_time = event["time_sec"]

        start_frame = round(event_time * fps) + 1

        if start_frame > scene.frame_end:
            continue

        event_intensity = event["intensity"] * max(0.0, intensity_scale)
        event_intensity = min(event_intensity, 2.0)

        # The height responds to event intensity.
        bounce_height = 0.75 * event_intensity
        peak_frame = start_frame + bounce_frames
        settle_frame = start_frame + settle_frames

        peak_frame = min(peak_frame, scene.frame_end)
        settle_frame = min(settle_frame, scene.frame_end)

        # Return to the normal position at the event start.
        cube.location = (0, 0, base_z)
        cube.scale = (1.0, 1.0, 1.0)

        cube.keyframe_insert(
            data_path="location",
            frame=start_frame,
            group="MusicMotion",
        )

        cube.keyframe_insert(
            data_path="scale",
            frame=start_frame,
            group="MusicMotion",
        )

        # Peak of the bounce.
        cube.location = (0, 0, base_z + bounce_height)

        stretch = min(1.0 + 0.08 * event_intensity, 1.22)
        cube.scale = (1.0 / math.sqrt(stretch),
                      1.0 / math.sqrt(stretch),
                      stretch)

        cube.keyframe_insert(
            data_path="location",
            frame=peak_frame,
            group="MusicMotion",
        )

        cube.keyframe_insert(
            data_path="scale",
            frame=peak_frame,
            group="MusicMotion",
        )

        # Return to the base pose.
        cube.location = (0, 0, base_z)
        cube.scale = (1.0, 1.0, 1.0)

        cube.keyframe_insert(
            data_path="location",
            frame=settle_frame,
            group="MusicMotion",
        )

        cube.keyframe_insert(
            data_path="scale",
            frame=settle_frame,
            group="MusicMotion",
        )

    # Smooth motion interpolation.
    if cube.animation_data and cube.animation_data.action:
        action = cube.animation_data.action

        # Blender 5.x uses layered actions. Handle available F-curves
        # without assuming the older action API.
        try:
            for layer in action.layers:
                for strip in layer.strips:
                    for slot in action.slots:
                        try:
                            channelbag = strip.channelbag(slot)
                            if channelbag is None:
                                continue

                            for fcurve in channelbag.fcurves:
                                for key in fcurve.keyframe_points:
                                    key.interpolation = "BEZIER"
                        except (AttributeError, TypeError, RuntimeError):
                            continue
        except (AttributeError, TypeError):
            pass

    # Camera
    bpy.ops.object.camera_add(location=(4.2, -7.5, 4.6))

    camera = bpy.context.object
    camera.name = "AnimationCamera"

    target = Vector((0, 0, 0.9))
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 50

    scene.camera = camera

    # Main area light
    bpy.ops.object.light_add(
        type="AREA",
        location=(1.5, -3.5, 6),
    )

    key_light = bpy.context.object
    key_light.name = "KeyLight"
    key_light.data.energy = 850
    key_light.data.shape = "DISK"
    key_light.data.size = 5

    light_direction = Vector((0, 0, 0.5)) - key_light.location
    key_light.rotation_euler = light_direction.to_track_quat(
        "-Z", "Y"
    ).to_euler()

    # Fill light
    bpy.ops.object.light_add(
        type="AREA",
        location=(-4, 1, 3.5),
    )

    fill_light = bpy.context.object
    fill_light.name = "FillLight"
    fill_light.data.energy = 350
    fill_light.data.size = 4

    fill_direction = Vector((0, 0, 0.5)) - fill_light.location
    fill_light.rotation_euler = fill_direction.to_track_quat(
        "-Z", "Y"
    ).to_euler()

    scene.frame_set(1)

    return scene


# ============================================================
# Video and audio composition
# ============================================================

def configure_video_output(scene, output_path):
    """Configure Blender's video encoder for MP4/H.264 and AAC."""
    scene.render.image_settings.media_type = "VIDEO"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.file_format = "FFMPEG"

    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"

    scene.render.ffmpeg.audio_codec = "AAC"
    scene.render.ffmpeg.audio_channels = "STEREO"
    scene.render.ffmpeg.audio_mixrate = 44100

    scene.render.filepath = str(output_path)
    scene.render.use_file_extension = True


def create_mix_scene(animation_scene, audio_path, duration):
    """
    Create a separate scene containing the animation scene strip
    and the music strip. This avoids adding the scene to itself.
    """
    mix_scene = bpy.data.scenes.new("MusicAnimationMix")

    mix_scene.render.engine = "BLENDER_EEVEE"
    mix_scene.render.resolution_x = 1280
    mix_scene.render.resolution_y = 720
    mix_scene.render.resolution_percentage = 100
    mix_scene.render.fps = 30
    mix_scene.render.fps_base = 1.0

    mix_scene.frame_start = 1
    mix_scene.frame_end = max(2, math.ceil(duration * 30) + 1)

    mix_scene.render.image_settings.color_mode = "RGB"

    # Important: the mix scene renders its Video Sequence Editor.
    mix_scene.render.use_sequencer = True

    editor = mix_scene.sequence_editor_create()

    # Put the rendered animation on channel 1.
    editor.strips.new_scene(
        name="AnimationVideo",
        scene=animation_scene,
        channel=1,
        frame_start=1,
    )

    # Put the WAV soundtrack on channel 2.
    sound_strip = editor.strips.new_sound(
        name="MusicAudio",
        filepath=str(audio_path),
        channel=2,
        frame_start=1,
    )

    sound_strip.volume = 1.0

    # Ensure the video and audio end at the same time.
    mix_scene.frame_set(1)

    return mix_scene


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments()

    events_path = Path(args.events).resolve()
    audio_path = Path(args.audio).resolve()
    output_path = Path(args.output).resolve()

    if not events_path.is_file():
        raise FileNotFoundError(f"Events CSV not found: {events_path}")

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(audio_path)
    events = load_events(events_path)

    speed = max(0.1, args.speed)
    intensity_scale = max(0.0, args.intensity)

    print("Audio:", audio_path, flush=True)
    print(f"Audio duration: {duration:.2f} seconds", flush=True)
    print(f"Beat events: {len(events)}", flush=True)
    print(f"Animation speed: {speed}", flush=True)
    print(f"Intensity scale: {intensity_scale}", flush=True)
    print("Output:", output_path, flush=True)

    # This script is intended to run in Blender's background
    # with --factory-startup, so it does not alter saved project files.
    animation_scene = bpy.context.scene
    animation_scene.name = "MusicAnimationScene"

    clear_scene(animation_scene)

    create_animation_scene(
        scene=animation_scene,
        duration=duration,
        events=events,
        speed=speed,
        intensity_scale=intensity_scale,
    )

    mix_scene = create_mix_scene(
        animation_scene=animation_scene,
        audio_path=audio_path,
        duration=duration,
    )

    configure_video_output(mix_scene, output_path)

    # Render from the composition scene so the final MP4 contains
    # both the animated scene and the WAV soundtrack.
    bpy.ops.render.render(
        animation=True,
        scene=mix_scene.name,
    )

    # Blender normally appends the extension automatically.
    expected_output = output_path

    if not expected_output.exists():
        alternative = Path(str(output_path) + ".mp4")

        if alternative.exists():
            expected_output = alternative

    if not expected_output.exists():
        raise RuntimeError(
            "Blender finished rendering but the expected MP4 was not found: "
            + str(output_path)
        )

    print("RENDER_SUCCESS:", expected_output, flush=True)
    print(
        f"Output size: {expected_output.stat().st_size / (1024 * 1024):.2f} MB",
        flush=True,
    )


if __name__ == "__main__":
    main()
