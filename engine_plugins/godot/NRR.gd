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


## Discards accumulated temporal history (scene cut / camera switch).
## Mirrors nrr_device_reset_temporal_history() from the public C API.
func reset_temporal_history() -> bool:
	if not available or _native == null:
		return false
	return bool(_native.call("reset_temporal_history"))


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
