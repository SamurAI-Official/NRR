"""Read back the probe's EXR passes. Separate from the probe on purpose: Blender 4.2.3 dies with an access
violation at the end of a background compositor render, after the File Output has already written the pass, so
the render and the read happen in different processes.

    blender --background --python tools/blender_capture/read_probe_exr.py
"""

import os
import statistics

import bpy

OUT_DIR = os.path.abspath(os.environ.get("NRR_BLENDER_PROBE_OUT", "blender_probe"))

print("[read] blender", bpy.app.version_string, "->", OUT_DIR)


def central(px, w, h, channel, frac=0.5):
    """Median of one channel over the central region.

    A camera pan gives the whole plane almost one vector, so a median over the middle is a robust summary of
    "what the field says" and is directly comparable to the projection ground truth. Sampling one pixel would
    make the answer sensitive to the checker pattern and to any edge.
    """
    x0, x1 = int(w * (0.5 - frac / 2)), int(w * (0.5 + frac / 2))
    y0, y1 = int(h * (0.5 - frac / 2)), int(h * (0.5 + frac / 2))
    vals = []
    for y in range(y0, y1):
        base = (h - 1 - y) * w
        for x in range(x0, x1):
            vals.append(px[(base + x) * 4 + channel])
    return statistics.median(vals), min(vals), max(vals)


for name in sorted(os.listdir(OUT_DIR)):
    if not name.endswith(".exr"):
        continue
    path = os.path.join(OUT_DIR, name)
    img = bpy.data.images.load(path)
    w, h = img.size
    px = list(img.pixels)
    print("[read] %s: %dx%d" % (name, w, h))
    for channel, label in ((0, "R"), (1, "G")):
        med, lo, hi = central(px, w, h, channel)
        print("[read]   %s median=%.6f min=%.6f max=%.6f" % (label, med, lo, hi))
    bpy.data.images.remove(img)
