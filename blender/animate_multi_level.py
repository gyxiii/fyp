
import csv
import math
from pathlib import Path

import bpy
from mathutils import Vector


PROJECT_DIR = Path(
    r"C:\Users\gyixu\Downloads\CAT404_music_animation_starter\CAT404_music_animation_starter"
)
EVENTS_PATH = PROJECT_DIR / "outputs" / "multi_level_events.csv"

FPS = 30
BASE_HEIGHT = 0.5
BOUNCE_HEIGHT = 1.0

# Maximum bounce duration in seconds.
MAX_BOUNCE_DURATION = 0.30


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def read_events(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find animation events file: {path}\n"
            "Run src/music_analysis.py and src/beat_baseline.py first."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    events = []
    for row in rows:
        if row.get("event_type") != "beat":
            continue

        time_sec = float(row["time_sec"])
        intensity = float(row.get("intensity", 1.0))

        if time_sec >= 0:
            events.append({
                "time": time_sec,
                "intensity": max(0.2, min(intensity, 1.5)),
            })

    return sorted(events, key=lambda event: event["time"])


def add_camera_and_light():
    bpy.ops.object.camera_add(location=(4, -7, 4))
    camera = bpy.context.object
    camera.name = "DemoCamera"

    direction = Vector((0, 0, 0.5)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(0, -4, 6))
    light = bpy.context.object
    light.name = "DemoLight"
    light.data.energy = 500
    light.data.shape = "DISK"
    light.data.size = 5


def get_bounce_durations(events):
    """Limit each bounce so it finishes before the next beat."""
    durations = []

    for index, event in enumerate(events):
        if index + 1 < len(events):
            gap = events[index + 1]["time"] - event["time"]
            duration = min(MAX_BOUNCE_DURATION, gap * 0.75)
        else:
            duration = MAX_BOUNCE_DURATION

        durations.append(max(0.05, duration))

    return durations


def calculate_height(time_sec, events, durations):
    """
    Calculate height from the active beat envelopes.

    Each beat produces one smooth, short bounce.
    The strongest active envelope determines the height.
    """
    bounce = 0.0

    for event, duration in zip(events, durations):
        elapsed = time_sec - event["time"]

        if 0.0 <= elapsed <= duration:
            progress = elapsed / duration

            # Smooth rise and fall, returning to the base height.
            envelope = math.sin(math.pi * progress)

            height = (
                BOUNCE_HEIGHT
                * event["intensity"]
                * envelope
            )

            bounce = max(bounce, height)

    return BASE_HEIGHT + bounce


def main():
    events = read_events(EVENTS_PATH)

    if not events:
        raise ValueError("No beat events found in animation_events.csv.")

    clear_scene()

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1

    scene.frame_end = max(
        FPS * 5,
        math.ceil((events[-1]["time"] + 1.0) * FPS),
    )

    bpy.ops.mesh.primitive_plane_add(
        size=10,
        location=(0, 0, 0),
    )
    floor = bpy.context.object
    floor.name = "Floor"

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(0, 0, BASE_HEIGHT),
    )
    cube = bpy.context.object
    cube.name = "BeatDrivenCube"

    add_camera_and_light()

    durations = get_bounce_durations(events)

    # Calculate a height for every frame.
    # This avoids conflicting keyframes from overlapping bounce clips.
    for frame in range(scene.frame_start, scene.frame_end + 1):
        time_sec = (frame - 1) / FPS
        cube.location.z = calculate_height(
            time_sec,
            events,
            durations,
        )
        cube.keyframe_insert(
            data_path="location",
            frame=frame,
            group="Music Driven Motion",
        )

    scene.frame_set(1)

    bpy.ops.object.select_all(action="DESELECT")
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube

    print("Beat-driven animation created successfully.")
    print(f"Loaded beat events: {len(events)}")
    print(f"Frame rate: {FPS} FPS")
    print(f"Animation ends at frame: {scene.frame_end}")
    print("Bounce timing follows beat timestamps.")
    print("Bounce height follows RMS-derived intensity.")


main()

