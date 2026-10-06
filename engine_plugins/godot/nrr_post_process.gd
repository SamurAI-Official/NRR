@tool
class_name NRRPostProcess
extends CanvasLayer

## Applies NRR to the rendered viewport, as a plain CanvasLayer overdraw.
##
## This is deliberately not a render-pipeline (CompositorEffect / RD) hook:
## a CanvasLayer + TextureRect works identically under forward_plus, mobile and
## gl_compatibility, so the same addon covers every hardware pipeline Godot can
## present with, including the GLES/Compatibility path that has no GPU compute
## hook at all.
##
## STATUS / LIMITS (read before enabling):
##  * When the native binding is absent this node degrades to a pure passthrough
##    that draws nothing extra - it never fakes neural output.
##  * Depth and motion vectors are NOT captured today. Godot exposes no portable
##    depth buffer to GDScript, and no motion-vector buffer at all outside
##    Forward+, so render_frame() is called with color only. NRR therefore runs
##    without its depth/motion conditioning inputs, which is a real limitation
##    tracked as M6/M7 in docs/roadmap.md.
##  * The viewport read-back is the previous completed frame, so the NRR output
##    is presented one frame late. Fine for an offline/quality pass, visible at
##    high frame rates.

signal stats_updated(stats: Dictionary)

@export var enabled: bool = true:
	set(value):
		enabled = value
		if _overlay != null:
			_overlay.visible = value

## Model to load. res:// paths are resolved for the native filesystem layer.
@export_file("*.onnx", "*.nrrmodel") var model_path: String = ""

## Cap the resolution the neural pass runs at, as a fraction of the viewport.
@export_range(0.25, 1.0, 0.05) var resolution_scale: float = 1.0

## Cross-fade between the original frame (0.0) and the NRR output (1.0).
@export_range(0.0, 1.0, 0.05) var blend_amount: float = 1.0

## When true, skips the neural pass and keeps the original frame.
@export var passthrough: bool = false

## Measure the camera's screen-space motion each frame and report it as the runtime's motion magnitude, so
## the history weight decays on a moving camera instead of ghosting. The measurement is the one both engine
## bindings use (NRR.measure_camera_motion): one world point at `motion_reference_depth` projected through
## the previous and current view-projections, over the frame width. It cannot see object motion.
@export var measure_motion: bool = true

## Depth of the measured reference point. Translation is exact at this depth and scales with the inverse of
## the real depth, so this should be the depth the subject mostly sits at.
@export_range(0.1, 200.0, 0.1) var motion_reference_depth: float = 4.0

## Overrides the measurement when non-zero - for a scene whose motion is known better than a camera can
## know it (object motion, or a subject off the reference depth).
@export var motion_magnitude_override: float = 0.0

## The sub-pixel sampling offset the renderer used for the frame being submitted, and whether the sequence
## jitters at all. A temporal upscaler already produces this per frame, so a caller driving DLSS/DLAA
## (Streamline, in the NVIDIA Godot fork) or TAA/FSR2 assigns it here; a caller that does not jitter leaves
## it disabled and the integration declines rather than averaging identically-phased frames.
@export var jitter_offset: Vector2 = Vector2.ZERO
@export var jitter_enabled: bool = false

## Optional per-pixel motion field for the submitted frame (RG half float, in pixels per frame, on the
## render grid), e.g. a DLSS velocity buffer. This is the field the phase-aligned pass warps by and the
## blend reprojects with; without it both fall back to the whole-frame magnitude.
@export var motion_vectors: Texture2D = null

var _nrr: NRR = null
var _overlay: TextureRect = null
var _output_texture: ImageTexture = null
var _ready_mode := false
var _warned_unavailable := false
var _frame_index := 0
## The camera state and jitter that belong to the frame being read back: the viewport read-back is the last
## *completed* frame, so what is submitted is one frame behind the camera right now, and reporting the
## current frame's motion or jitter for it would misdescribe it.
var _previous_camera_transform := Transform3D()
var _previous_camera_projection := Projection()
var _has_previous_camera := false
var _submitted_jitter := Vector2.ZERO


func _ready() -> void:
	_ready_mode = Engine.is_editor_hint()
	if _ready_mode and not enabled:
		return

	_nrr = NRR.new()
	if not _nrr.initialize():
		if not _warned_unavailable:
			_warned_unavailable = true
			push_warning(
				"NRRPostProcess: native NRR binding unavailable (%s); running as "
				% _nrr.last_error
				+ "passthrough. Build engine_plugins/godot/src for this platform "
				+ "to enable neural rendering.")
	else:
		_nrr.load_model(model_path)
		# Pay the one-time neural-stack initialization here, at the size _process()
		# will actually render at, rather than stalling the first presented frame.
		# Measured: 632 ms cold against 11.7 ms steady state on a CUDA host.
		if _nrr.loaded_model != "":
			var warm_size := _render_size()
			_nrr.warmup(3, warm_size.x, warm_size.y)

	_build_overlay()


func _exit_tree() -> void:
	if _nrr != null:
		_nrr.shutdown()


func _process(delta: float) -> void:
	if not enabled or _overlay == null:
		return
	if _ready_mode:
		return
	if passthrough or _nrr == null or not _nrr.available or _nrr.loaded_model == "":
		return

	var source := get_viewport().get_texture()
	if source == null:
		return
	var frame := source.get_image()
	if frame == null:
		return
	frame.convert(Image.FORMAT_RGBA8)

	if resolution_scale < 1.0:
		frame.resize(
			maxi(1, int(frame.get_width() * resolution_scale)),
			maxi(1, int(frame.get_height() * resolution_scale)),
			Image.INTERPOLATE_BILINEAR)

	# What the caller told the runtime about the frame being submitted: the offset the renderer sampled it
	# at, and how far the scene moved. Both are *lagged* by one frame on purpose - the read-back is the last
	# completed frame, so the camera state and jitter that belong to it are the previous ones, and reporting
	# this frame's would misdescribe the frame the runtime is actually being handed.
	_nrr.set_jitter(_submitted_jitter, jitter_enabled)
	var magnitude := motion_magnitude_override
	if magnitude <= 0.0 and measure_motion:
		magnitude = NRR.measure_camera_motion(
				get_viewport().get_camera_3d(), _previous_camera_transform,
				_previous_camera_projection, float(frame.get_width()),
				motion_reference_depth, _has_previous_camera)
	if magnitude > 0.0:
		_nrr.set_motion_magnitude(magnitude)
	_submitted_jitter = jitter_offset if jitter_enabled else Vector2.ZERO
	var camera := get_viewport().get_camera_3d()
	if camera != null:
		_previous_camera_transform = camera.global_transform
		_previous_camera_projection = camera.get_camera_projection()
		_has_previous_camera = true

	var motion_image: Image = null
	if motion_vectors != null:
		motion_image = motion_vectors.get_image()
		if motion_image != null:
			motion_image.convert(Image.FORMAT_RGH)

	var out := _nrr.render_frame(frame, null, motion_image)
	if _nrr.last_render_was_passthrough:
		stats_updated.emit({"frame_index": _frame_index, "passthrough": true})
		return

	if _output_texture == null \
			or _output_texture.get_width() != out.get_width() \
			or _output_texture.get_height() != out.get_height():
		_output_texture = ImageTexture.create_from_image(out)
		_apply_output_texture()
	else:
		_output_texture.update(out)

	_frame_index += 1
	var stats := {
		"frame_index": _frame_index - 1,
		"passthrough": false,
		"backend": _nrr.backend_name(),
		"render_time_ms": _nrr.last_render_time_ms(),
	}
	# What the runtime decided, so a caller can tell "the integration ran" from "it declined" from "it is
	# off" - which is the difference between a working temporal path and one that looks configured.
	stats.merge(_nrr.temporal_state(), true)
	stats_updated.emit(stats)


func _render_size() -> Vector2i:
	# The size _process() will submit, so the warm-up covers the shape that matters.
	var vp := get_viewport()
	if vp == null:
		return Vector2i(64, 48)
	var size := vp.get_visible_rect().size
	return Vector2i(maxi(1, int(size.x * resolution_scale)),
			maxi(1, int(size.y * resolution_scale)))


func _build_overlay() -> void:
	_overlay = TextureRect.new()
	_overlay.name = "NRROverlay"
	_overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
	_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_overlay.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_overlay.stretch_mode = TextureRect.STRETCH_SCALE
	var material := ShaderMaterial.new()
	var shader: Shader = load("res://addons/nrr/shaders/nrr_blit.gdshader")
	if shader != null:
		material.shader = shader
	material.set_shader_parameter("blend_amount", blend_amount)
	_overlay.material = material
	_overlay.visible = enabled
	add_child(_overlay)
	_apply_output_texture()


func _apply_output_texture() -> void:
	if _overlay == null:
		return
	_overlay.texture = _output_texture
	var material := _overlay.material as ShaderMaterial
	if material != null:
		material.set_shader_parameter("nrr_output", _output_texture)
		material.set_shader_parameter("blend_amount", blend_amount)
