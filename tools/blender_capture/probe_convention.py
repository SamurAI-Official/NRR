"""Probe Blender's motion-vector and depth conventions *before* capturing any training data.

Why a probe rather than a capture: this is the same step the Godot capture needed
(`tools/godot_capture/`), and for the same reason - a convention inferred from documentation is how a
comparison ends up measuring the wrong thing. The Godot path pins its motion vectors against the matrices a
renderer uses; here the ground truth comes from Blender's own projection,
`bpy_extras.object_utils.world_to_camera_view`, which is computed independently of the render. So the
rendered Vector pass is compared against a number that is *not* derived from it, and the sign, the scale
(pixels or normalised) and the Y direction are measured rather than assumed.

What it does, per engine:
  1. builds a scene whose screen-space displacement is known analytically (a plane at a fixed depth and a
     cube, camera translated by exactly N internal-grid pixels),
  2. asks for the Z (depth) and Vector (motion) passes,
  3. renders frames 1 and 2 with single-layer EXRs out of the compositor,
  4. reads the EXRs back and prints the measured value at known pixels next to the projection ground truth,
  5. reports which passes produced data at all - a pass that is silently empty is a finding, not a failure.

Run headless:
  blender --background --python tools/blender_capture/probe_convention.py
Outputs go to $NRR_BLENDER_PROBE_OUT (default ./blender_probe): one EXR per pass plus summary.json.
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

RES = (128, 96)          # the internal (render) grid a capture would use
LENS_MM = 50.0
SENSOR_MM = 36.0
DEPTH_M = 4.0            # camera distance to the reference plane
SHIFT_PX = 2.0           # the exact screen-space shift the camera translation should produce
OUT_DIR = os.path.abspath(os.environ.get("NRR_BLENDER_PROBE_OUT", "blender_probe"))


def log(*args):
    print("[probe]", *args, flush=True)


# --- scene -----------------------------------------------------------------------------------------

def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_scene():
    """A checker-textured plane at -DEPTH_M and a cube off-centre, lit by a sun.

    The plane carries a checker *texture* rather than a flat colour: a flat surface would make the low-res
    pass identical everywhere and hide a sub-pixel error, which is the trap the Godot fixtures avoid too.
    """
    scene = bpy.context.scene
    scene.render.resolution_x = RES[0]
    scene.render.resolution_y = RES[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("ProbeWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.04, 0.04, 0.05, 1.0)

    # Plane, 32 m across, facing +Z.
    bpy.ops.mesh.primitive_plane_add(size=32.0, location=(0.0, 0.0, -DEPTH_M))
    plane = bpy.context.active_object
    plane.name = "ProbePlane"
    mat = bpy.data.materials.new("Checker")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 40.0
    checker.inputs["Color1"].default_value = (0.85, 0.85, 0.85, 1.0)
    checker.inputs["Color2"].default_value = (0.08, 0.08, 0.08, 1.0)
    links.new(checker.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    plane.data.materials.append(mat)

    # A cube near the plane's centre, so the depth pass has two distinct surfaces to report.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -DEPTH_M + 0.5))
    cube = bpy.context.active_object
    cube.name = "ProbeCube"

    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("Sun", light_data)
    light.rotation_euler = (math.radians(35.0), math.radians(15.0), math.radians(25.0))
    scene.collection.objects.link(light)

    cam_data = bpy.data.cameras.new("ProbeCam")
    cam_data.lens = LENS_MM
    cam_data.sensor_width = SENSOR_MM
    cam = bpy.data.objects.new("ProbeCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_euler = (0.0, 0.0, 0.0)     # default orientation looks down -Z, which is what we want
    return scene, cam


def shift_for_pixels(px):
    """World translation, perpendicular to the view axis, that moves content by `px` at DEPTH_M.

    A perspective camera's focal length in pixels is res_x * lens / sensor_width, so a point at distance d
    shifts by dx_world * f_px / d. Inverting that gives the translation for a wanted pixel shift - the
    camera moves the other way, which is why the sign is applied by the caller.
    """
    f_px = RES[0] * LENS_MM / SENSOR_MM
    return px * DEPTH_M / f_px


def keyframe_motion(scene, cam, dx_world, axis="x"):
    """Key the camera's translation on frame 2, in one axis.

    The axis is a parameter because the sign of the vertical component has to be measured separately: a
    horizontal pan has a zero G in the Vector pass, so it cannot say anything about whether Y is mirrored -
    and a mirrored Y is the classic way a motion field looks plausible and is wrong.
    """
    for frame, offset in ((1, 0.0), (2, -dx_world)):
        scene.frame_set(frame)
        if axis == "y":
            cam.location = (0.0, offset, 0.0)
        else:
            cam.location = (offset, 0.0, 0.0)
        cam.keyframe_insert(data_path="location", frame=frame)


# --- projection ground truth ------------------------------------------------------------------------

def projection_truth(scene, cam, world_point):
    """Where Blender itself says a world point lands, in internal-grid pixels, at each frame.

    This is the probe's ground truth and it does not involve the renderer's passes at all: the difference
    between the two frames is what a *correct* motion field must report for that point.
    """
    out = {}
    for frame in (1, 2):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        ndc = world_to_camera_view(scene, cam, world_point)
        out[frame] = (ndc.x * RES[0], (1.0 - ndc.y) * RES[1], ndc.z)
    delta = (out[2][0] - out[1][0], out[2][1] - out[1][1])
    return {"frame1_px": out[1], "frame2_px": out[2], "delta_px": delta}


# --- passes -----------------------------------------------------------------------------------------

def enable_passes(view_layer, want_vector, want_z):
    """Ask for the motion and depth passes, reporting refusal instead of raising.

    A pass an engine does not have is the finding this probe exists to produce: "this engine cannot give you
    motion vectors" changes what the capture pipeline must be, and it is far cheaper to learn here than after
    a scene library has been built around the assumption.
    """
    vector_ok = z_ok = False
    try:
        view_layer.use_pass_vector = want_vector
        vector_ok = bool(getattr(view_layer, "use_pass_vector", False))
    except (AttributeError, TypeError):
        pass
    try:
        view_layer.use_pass_z = want_z
        z_ok = bool(getattr(view_layer, "use_pass_z", False))
    except (AttributeError, TypeError):
        pass
    return vector_ok, z_ok


def link_pass_output(scene, socket_name, tag):
    """Wire one Render Layers socket to a File Output node writing a single-layer EXR.

    Single-layer deliberately: a multilayer EXR needs layer naming rules to read back, and this probe wants one
    number per pass.
    """
    tree = scene.node_tree
    rl = next((n for n in tree.nodes if n.type == "R_LAYERS"), None)
    if rl is None or socket_name not in rl.outputs:
        return None
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = OUT_DIR
    out.format.file_format = "OPEN_EXR"
    out.format.color_depth = "32"
    out.file_slots[0].path = tag
    tree.links.new(rl.outputs[socket_name], out.inputs[0])
    return out


def render_frame(scene, frame):
    scene.frame_set(frame)
    bpy.ops.render.render(write_still=False)


def read_exr(path):
    img = bpy.data.images.load(path)
    size = img.size
    pixels = list(img.pixels)
    bpy.data.images.remove(img)
    return size[0], size[1], pixels


def sample(pixels, w, h, x, y):
    """RGBA at a pixel, converting from the top-down coordinates used everywhere else.

    Blender's pixel buffer is bottom-up, which is a classic source of a motion field that looks plausible and
    is vertically mirrored.
    """
    px = max(0, min(w - 1, int(round(x))))
    py = max(0, min(h - 1, int(round(y))))
    i = ((h - 1 - py) * w + px) * 4
    return [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]


def newest_exr(tag):
    names = [n for n in sorted(os.listdir(OUT_DIR)) if n.startswith(tag) and n.endswith(".exr")]
    return os.path.join(OUT_DIR, names[-1]) if names else None


# --- the probe itself ---------------------------------------------------------------------------------

def parse_args():
    """Options after `--`, so this can be driven one combination per Blender invocation."""
    argv = sys.argv
    if "--" not in argv:
        return {}
    rest = argv[argv.index("--") + 1:]
    return {rest[i].lstrip("-"): rest[i + 1] for i in range(0, len(rest) - 1, 2)}


def flush(engine, pass_name, axis, entry, suffix=""):
    """Write the entry to JSON now, with a suffix so a later full write cannot hide an early one."""
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "%s_%s_%s%s.json" % (engine, pass_name, axis, suffix)
    with open(os.path.join(OUT_DIR, name), "w") as handle:
        json.dump({"blender": bpy.app.version_string, "result": entry}, handle, indent=2, default=str)
    return name


def probe_one(engine, pass_name, entry, axis="x"):
    """Render exactly one pass, once, in this process.

    One pass per invocation is not tidiness: with two File Output nodes wired and a second render, Blender
    4.2.3 dies with an access violation during compositing *after* the first pass has been written - so the
    driver runs each combination in its own process and a crash costs one combination instead of the probe.
    """
    clean_scene()
    scene, cam = build_scene()
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = 16
        scene.cycles.use_denoising = False
        # Cycles computes the Vector pass only when motion blur is on. The shutter interval decides *which*
        # motion it measures, but measured in Blender 4.2.3 it is not the whole story: with the pass written
        # through a compositor File Output node, one run produced a correct field (2.0 px on the plane, 2.667 on
        # the cube's face - exactly the 4/depth ratio) and later runs with identical settings produced zeros,
        # while the same process crashes with an access violation during compositing. See README.md: the
        # recommendation is not to build the capture path on this pass. It is also why a capture has to render
        # the beauty and the vectors in two separate renders - with blur on, the beauty is blurred.
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = 1.0
        scene.render.motion_blur_position = "END"

    dx = shift_for_pixels(SHIFT_PX)
    keyframe_motion(scene, cam, dx, axis)
    entry["camera_dx_world"] = dx
    entry["axis"] = axis
    # The evaluated pose at the frame being rendered, recorded before the render: a keyframe that did not take
    # effect shows up here as the origin, which is the difference between "no motion in the field" and "no
    # motion in the scene".
    scene.frame_set(2)
    bpy.context.view_layer.update()
    entry["camera_location_frame2"] = [round(v, 6) for v in cam.matrix_world.translation]

    plane_point = Vector((0.0, 1.5, -DEPTH_M))                # on the plane, off-centre, clear of the cube
    cube_point = Vector((0.0, 0.0, -DEPTH_M + 0.5))            # the cube's centre: 0.5 m nearer, so parallax
    entry["truth_plane"] = projection_truth(scene, cam, plane_point)
    entry["truth_cube"] = projection_truth(scene, cam, cube_point)

    vector_ok, z_ok = enable_passes(bpy.context.view_layer, True, True)
    entry["pass_vector_enabled"] = vector_ok
    entry["pass_z_enabled"] = z_ok

    socket = "Depth" if pass_name == "z" else "Vector"
    tag = "%s_%s" % (pass_name, axis)
    scene.use_nodes = True
    for node in list(scene.node_tree.nodes):
        scene.node_tree.nodes.remove(node)
    scene.node_tree.nodes.new("CompositorNodeRLayers")
    if link_pass_output(scene, socket, tag) is None:
        entry["socket"] = "absent"
        return
    entry["socket"] = "present"

    # Flush before rendering: the render is the call that kills Blender 4.2.3 in background mode, so anything
    # not written first is lost. The ground truth is what makes the measured values interpretable.
    flush(engine, pass_name, axis, entry, suffix="_pre")
    render_frame(scene, 2)
    path = newest_exr(tag)
    if path is None:
        entry["file"] = "no file written"
        return
    entry["file"] = os.path.basename(path)
    w, h, pixels = read_exr(path)
    entry["size"] = [w, h]
    for name, truth in (("plane", entry["truth_plane"]), ("cube", entry["truth_cube"])):
        x, y = truth["frame2_px"][0], truth["frame2_px"][1]
        entry[name] = {
            "at_px": [round(x, 2), round(y, 2)],
            "rgba": [round(v, 6) for v in sample(pixels, w, h, x, y)],
        }

    # What the measured field *means*, as a ratio against the projection ground truth rather than an opinion: a
    # ratio of +1 is "current minus previous, in pixels on the internal grid", -1 is the opposite sense, and a
    # magnitude far from 1 says the units are normalised rather than pixels.
    if pass_name == "vec":
        truth = entry["truth_plane"]["delta_px"]
        got = entry["plane"]["rgba"]
        entry["verdict"] = {
            "truth_delta_px": [round(v, 4) for v in truth],
            "measured_xy": [round(got[0], 6), round(got[1], 6)],
            "ratio_x": round(got[0] / truth[0], 4) if truth[0] else None,
            "ratio_y": round(got[1] / truth[1], 4) if truth[1] else None,
            "same_direction": (got[0] * truth[0] + got[1] * truth[1]) > 0.0,
        }


def report_only():
    """No options: report which passes each engine will accept, without rendering anything."""
    summary = {"blender": bpy.app.version_string, "mode": "report_only", "engines": {}}
    for engine in ("CYCLES", "BLENDER_EEVEE_NEXT"):
        try:
            clean_scene()
            scene, _cam = build_scene()
            scene.render.engine = engine
            vector_ok, z_ok = enable_passes(bpy.context.view_layer, True, True)
            summary["engines"][engine] = {"pass_vector_enabled": vector_ok, "pass_z_enabled": z_ok}
        except Exception as exc:
            summary["engines"][engine] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = parse_args()
    if "engine" not in args:
        summary = report_only()
        log(json.dumps(summary, indent=2, default=str))
    else:
        engine = args["engine"].upper()
        pass_name = args.get("pass", "vec")
        axis = args.get("axis", "x")
        entry = {"engine": engine, "pass": pass_name, "axis": axis}
        try:
            probe_one(engine, pass_name, entry, axis)
        except Exception as exc:
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
        log(engine, pass_name, axis, json.dumps(entry, indent=2, default=str))
        with open(os.path.join(OUT_DIR, "%s_%s_%s.json" % (engine, pass_name, axis)), "w") as handle:
            json.dump({"blender": bpy.app.version_string, "result": entry}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()



