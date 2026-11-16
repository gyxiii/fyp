"""Extract a frame from a rendered MP4 using Blender's sequencer (QA helper).

Usage:
    blender --background --factory-startup --python tools/extract_frame.py -- <video> <out.png> <frame>
"""

import sys

import bpy

argv = sys.argv
argv = argv[argv.index("--") + 1:]
video, output, frame = argv[0], argv[1], int(argv[2])

scene = bpy.context.scene
scene.sequence_editor_create()
scene.sequence_editor.strips.new_movie(
    name="clip", filepath=video, channel=1, frame_start=1
)
scene.render.use_sequencer = True
scene.render.image_settings.file_format = "PNG"
scene.render.resolution_percentage = 100
scene.frame_set(frame)
scene.render.filepath = output
bpy.ops.render.render(write_still=True)
print("FRAME_SAVED", output)
