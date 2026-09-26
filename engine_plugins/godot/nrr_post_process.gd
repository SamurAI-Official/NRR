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

var _nrr: NRR = null
var _overlay: TextureRect = null
var _output_texture: ImageTexture = null
var _ready_mode := false
var _warned_unavailable := false
var _frame_index := 0


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

	# Depth and motion are intentionally omitted - see the header note.
	var out := _nrr.render_frame(frame, null, null)
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
	stats_updated.emit({
		"frame_index": _frame_index - 1,
		"passthrough": false,
		"backend": _nrr.backend_name(),
		"render_time_ms": _nrr.last_render_time_ms(),
	})


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
