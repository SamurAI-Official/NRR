@tool
extends RefCounted
class_name NRR

## NRR (Neural Rendering Runtime) - Godot 4.x GDScript API.
##
## Proxies the portable NRR C API (specification/api.md) through the
## GDExtension registered in nrr.gdextension (class `NRRNative`).
##
## HONEST AVAILABILITY: the native binding is a per-platform build output and
## is not committed to the NRR repository. When it is missing, or was built for
## a different platform, `initialize()` returns false, `available` stays false,
## and `render_frame()` returns the color image UNCHANGED. This class never
## fabricates neural output; callers MUST treat a returned image as passthrough
## unless `last_render_was_passthrough` is false. Callers that need to branch
## should test `available` rather than trusting that a render "happened".
##
## The method surface matches ShugoCore's Godot binding
## (platforms/godot/addons/nrr_godot/NRR.gd) so one game can move between the
## contract stub and this real binding without changing call sites.

signal model_loaded(path: String)
signal model_unloaded
signal render_completed(image: Image)

## Name of the native class the GDExtension registers.
const NATIVE_CLASS := "NRRNative"

## Native handle, or null when the extension is absent.
var available: bool = false

## Path of the currently loaded model ("" when none).
var loaded_model: String = ""

## True when the last render_frame() call returned its input untouched.
var last_render_was_passthrough: bool = true

## Last error reported by the native layer ("" when none).
var last_error: String = ""

var _native: Object = null


## Loads the native binding and creates an NRR device. Returns true only when
## the runtime is genuinely usable on this platform.
func initialize() -> bool:
	last_error = ""
	_native = _instantiate_native()
	if _native == null:
		available = false
		last_error = "NRR GDExtension not loaded for this platform"
		push_warning("NRR: %s" % last_error)
		return false

	var ok: bool = bool(_native.call("initialize"))
	available = ok
	if not ok:
		last_error = str(_native.call("get_last_error"))
		push_warning("NRR: native initialize() failed: %s" % last_error)
	return ok


func shutdown() -> void:
	if _native != null and available:
		_native.call("shutdown")
	available = false
	loaded_model = ""
	_native = null


## Loads a model by path. Accepts res:// paths; they are resolved to an OS
## path because the C API opens files through the platform filesystem.
func load_model(path: String) -> bool:
	if not available or _native == null:
		return false
	var os_path := ProjectSettings.globalize_path(path)
	var ok: bool = bool(_native.call("load_model", os_path))
	if ok:
		loaded_model = path
		model_loaded.emit(path)
	else:
		last_error = str(_native.call("get_last_error"))
	return ok


func unload_model() -> bool:
	if not available or _native == null:
		return false
	var ok: bool = bool(_native.call("unload_model"))
	if ok:
		loaded_model = ""
		model_unloaded.emit()
	return ok


## Renders one frame.
##
## color/depth/motion must be the same size and share a pixel format the C API
## understands (RGBA8 for color, R32F for depth, RG16F for motion). Passing
## null for depth or motion is allowed: the corresponding conditioning input is
## then omitted, which the runtime reports through its capability block.
func render_frame(color: Image, depth: Image = null, motion: Image = null) -> Image:
	last_render_was_passthrough = true
	if not available or _native == null or color == null:
		if color == null:
			last_error = "render_frame requires a color image"
		return color
	if loaded_model == "":
		last_error = "no model loaded; call load_model() first"
		return color

	var out: Variant = _native.call("render_frame", color, depth, motion)
	if out is Image:
		last_render_was_passthrough = color == out
		last_error = str(_native.call("get_last_error"))
		var result: Image = out
		render_completed.emit(result)
		return result
	last_error = "native render_frame returned no image"
	return color


## Renders `frames` throwaway frames and discards them, so the one-time cost of
## bringing up the neural stack (CUDA/cuDNN context creation, kernel autotuning,
## ONNX Runtime session warm-up) is paid before the first frame anyone sees.
## Returns true when the warm-up produced a real render.
##
## HONESTY: a first frame measured without this is dominated by initialization,
## not by per-frame cost. 632 ms cold against 11.7 ms steady state was measured
## for a 512x512 model on a CUDA host (M2, docs/roadmap.md). Report the first
## frame and the steady state separately; never present the cold number as the
## render cost. `width`/`height` should match the size you will actually render
## at, because providers do per-shape work (cuDNN algorithm selection) on first
## use, so a 64x48 warm-up does not fully cover a 1920x1080 frame.
func warmup(frames: int = 3, width: int = 64, height: int = 48) -> bool:
	if not available or _native == null or loaded_model == "":
		return false
	if frames <= 0 or width < 1 or height < 1:
		return false
	var probe := Image.create_empty(width, height, false, Image.FORMAT_RGBA8)
	probe.fill(Color(0.5, 0.5, 0.5, 1.0))
	var rendered := false
	for i in frames:
		if render_frame(probe) == null:
			return false
		rendered = not last_render_was_passthrough
	return rendered


## Discards accumulated temporal history (scene cut / camera switch).
## Mirrors nrr_device_reset_temporal_history() from the public C API.
func reset_temporal_history() -> bool:
	if not available or _native == null:
		return false
	return bool(_native.call("reset_temporal_history"))


## Integrates the distinct sub-pixel samples of several frames into one displayed frame.
## Mirrors nrr_device_set_phase_aligned_accumulation() from the public C API.
##
## Two things this binding does not supply today, and the runtime needs both: the per-frame sub-pixel
## offset (render_frame() submits frames with no jitter, so the runtime declines to integrate - it will
## not average identically-phased frames and call it antialiasing) and a per-frame motion measurement
## (motion_magnitude; the runtime stops integrating once a frame's scene motion exceeds 0.2 px, and a zero
## passes that gate on every frame). The magnitude a caller fills in is the camera's screen-space movement
## as a fraction of the frame width: project one world point at a reference depth through the previous and
## the current camera matrices, take the distance it moved in pixels, divide by the frame width. That is
## exact for rotation at any depth and exact for translation at the reference depth - the Unity renderer
## does exactly this (NRRRenderer.MeasureMotion) and its camera test measures the result against a render.
## The switch is exposed so a caller that supplies both through the native interface can use it, and so its
## state is at least observable from GDScript.
## Returns false when the device is missing or its backend has no accumulator (see last_error()).
func set_phase_aligned_accumulation(enabled: bool) -> bool:
	if not available or _native == null:
		return false
	return bool(_native.call("set_phase_aligned_accumulation", enabled))


## 1 when the accumulator has the integration on, 0 when it is off, -1 when this device cannot
## integrate at all - "off" and "cannot" are different answers, and only one of them means the
## runtime is doing what was asked.
func phase_aligned_accumulation() -> int:
	if not available or _native == null:
		return -1
	return int(_native.call("get_phase_aligned_accumulation"))


## Reports the scene motion the runtime should assume for the frames submitted next, in the unit
## NRRFrameInput::temporal.motion_magnitude is declared in: pixels moved per frame divided by the frame
## width (specification/frame_contract.md 4.3). Zero means "no measurement", which makes the
## phase-aligned gate fail closed rather than integrate a moving scene; the runtime clamps to [0, 1].
## See measure_camera_motion() for the measurement both engine bindings use.
func set_motion_magnitude(value: float) -> bool:
	if not available or _native == null:
		return false
	if not _native.has_method("set_motion_magnitude"):
		# An older extension: the method is not there yet. Reported rather than silently dropped, because a
		# benchmark that cannot tell "not supported" from "set to zero" is how a temporal path looks
		# configured while doing nothing.
		last_error = "extension has no set_motion_magnitude(); rebuild engine_plugins/godot/src"
		return false
	return bool(_native.call("set_motion_magnitude", value))


## The value the caller last set (the runtime's *used* value is in temporal_state()).
func motion_magnitude() -> float:
	if _native == null:
		return 0.0
	return float(_native.call("get_motion_magnitude"))


## The sub-pixel offset the renderer sampled this frame's grid at, and whether the sequence jitters.
##
## Without this the runtime declines to integrate: it will not average identically-phased frames and call
## it antialiasing, which is the honest answer but not a working feature. A temporal upscaler already
## produces this offset every frame - DLSS/DLAA through Streamline in the NVIDIA Godot fork
## (RendererRD::DLSSContext::Parameters::jitter), TAA or FSR2 otherwise - so a caller driving one of those
## hands it straight through rather than inventing a sequence.
func set_jitter(offset: Vector2, enabled: bool = true) -> bool:
	if not available or _native == null:
		return false
	if not _native.has_method("set_jitter"):
		last_error = "extension has no set_jitter(); rebuild engine_plugins/godot/src"
		return false
	return bool(_native.call("set_jitter", offset, enabled))


## The sub-pixel offsets a temporal upscaler produces, without which the integration declines.
func has_temporal_inputs() -> bool:
	if _native == null:
		return false
	return _native.has_method("set_jitter") and _native.has_method("set_motion_magnitude")


## What the runtime decided about the last frame, as a Dictionary: `motion_magnitude`, `temporal_alpha`,
## `history_frames`, `jitter_enabled`, `phase_aligned_enabled`, `frame_index` and `debug_info` (the
## runtime's own note, which distinguishes "off", "declined" and "ran"). Empty when nothing has rendered.
func temporal_state() -> Dictionary:
	if _native == null:
		return {}
	var value: Variant = _native.call("get_temporal_state")
	return value if value is Dictionary else {}


## The camera's screen-space motion this frame, as a fraction of the frame width - the measurement the
## Unity renderer performs (NRRRenderer.MeasureMotion) and the reference Godot binding documents.
##
## One world point at `reference_depth` in front of the camera is projected through the previous and the
## current view-projection matrices; the distance it appears to move on screen, over the frame width, is
## the scene motion. It is exact for rotation at any depth (the point's depth cancels) and exact for
## translation at the reference depth, scaling with the inverse of the real depth otherwise. Object motion
## is invisible to it, which is exactly what a camera can be expected to know.
##
## Returns 0.0 when there is no camera or no previous transform to compare against (the first frame).
static func measure_camera_motion(camera: Camera3D, previous_transform: Transform3D,
		previous_projection: Projection, frame_width: float, reference_depth: float,
		previous_visible: bool) -> float:
	if camera == null or not previous_visible or frame_width <= 0.0 or reference_depth <= 0.0:
		return 0.0
	# The same world point, projected through both frames' view-projections and compared on screen. The
	# previous *projection* is a parameter rather than re-read from the camera: an FOV change or a resize
	# between frames is exactly the kind of thing that would otherwise be charged to motion.
	var world := camera.global_transform * Vector3(0.0, 0.0, -reference_depth)
	var current := _project(world, camera.get_camera_projection(), camera.global_transform)
	var previous := _project(world, previous_projection, previous_transform)
	var moved := Vector2((current.x - previous.x) * 0.5 * frame_width,
			(current.y - previous.y) * 0.5 * frame_width)
	return moved.length() / frame_width


## View-projection of a world point, in normalised device coordinates. Kept separate so the arithmetic is
## testable without a renderer. Two GDScript facts shape it: there is no `Projection * Vector3` (so the clip
## space vector is a Vector4, divided by w here - the step a GPU performs), and `Projection(Transform3D)` is
## not exposed, so the view transform is applied on its own and the projection follows.
static func _project(world: Vector3, projection: Projection, camera_transform: Transform3D) -> Vector3:
	var view := camera_transform.affine_inverse() * world
	var clip := projection * Vector4(view.x, view.y, view.z, 1.0)
	if absf(clip.w) < 1e-9:
		return Vector3.ZERO
	return Vector3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)


## Name of the backend the device actually selected, e.g. "CPU".
func backend_name() -> String:
	if not available or _native == null:
		return ""
	return str(_native.call("get_backend_name"))


## Device capability block as a Dictionary (see NRRCapabilities in nrr.h).
func capabilities() -> Dictionary:
	if not available or _native == null:
		return {}
	var value: Variant = _native.call("get_capabilities")
	return value if value is Dictionary else {}


## Measured timings for the last render, in milliseconds.
func last_render_time_ms() -> float:
	if not available or _native == null:
		return 0.0
	return float(_native.call("get_render_time_ms"))


func native_entry_point_count() -> int:
	var native := _native_if_available()
	if native == null:
		return 0
	return int(native.call("get_entry_point_count"))


## Compile-time library version string ("" only when the binding is genuinely
## absent - this probes the class rather than reporting on initialize() state).
func library_version() -> String:
	var native := _native_if_available()
	if native == null:
		return ""
	return str(native.call("get_library_version"))


## The loaded model's session metadata as JSON, including the execution provider
## ONNX Runtime actually attached ("CPUExecutionProvider" or
## "CUDAExecutionProvider"). "" when no model is loaded.
func model_info() -> String:
	if _native == null:
		return ""
	return str(_native.call("get_model_info"))


## True when the GDExtension is loadable on this platform, independent of
## whether initialize() has been called. Use this to tell "the binding is
## missing" apart from "initialize() has not run yet".
func is_binding_present() -> bool:
	return _native_if_available() != null


## The native handle, instantiating a probe instance when none exists yet.
## Probing is safe: NRRNative allocates no device until initialize().
func _native_if_available() -> Object:
	if _native != null:
		return _native
	return _instantiate_native()


func _instantiate_native() -> Object:
	if not ClassDB.class_exists(NATIVE_CLASS):
		return null
	var value: Variant = ClassDB.instantiate(NATIVE_CLASS)
	if value is Object:
		return value
	return null
