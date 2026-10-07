extends SceneTree

# Probes the engine-side bridge (engine_plugins/godot_fork_patch) from GDScript.
#
#   <godot> --headless --path <this project> --script res://probe_bridge.gd
#
# Three states must be distinguishable, and this prints which one it is rather than a number that could be
# read as measurement:
#   * no hatch at all      -> Engine.has_singleton("NRRDLSS") is false  (stock Godot)
#   * hatch, DLSS idle     -> present, is_available() true, has_last_frame() false
#   * DLSS has run         -> has_last_frame() true, with a jitter and a velocity image
#
# Exit code: 0 when the answer is coherent (including "absent", which is a correct answer on stock Godot),
# 2 when the bridge contradicts itself (present but reporting a jitter without a frame).

func _initialize() -> void:
	print("=== NRR bridge probe ===")
	var info := Engine.get_version_info()
	print("godot=%s" % info.string)
	print("video_adapter=%s" % RenderingServer.get_video_adapter_name())
	print("engine.has_singleton(NRRDLSS)=%s" % Engine.has_singleton("NRRDLSS"))

	var bridge: Object = Engine.get_singleton("NRRDLSS")
	if bridge == null:
		print("bridge: ABSENT (this engine build has no DLSS effect to read, or the module is not built in)")
		print("RESULT: ABSENT")
		quit(0)
		return

	var status: Dictionary = bridge.call("status")
	print("bridge.status=%s" % JSON.stringify(status, "  "))
	var has_frame: bool = bool(bridge.call("has_last_frame"))
	var jitter: Vector2 = bridge.call("jitter")
	if not has_frame:
		var zero_ok: bool = jitter == Vector2.ZERO
		print("bridge: PRESENT, DLSS has not evaluated a frame (jitter is zero: %s)" % zero_ok)
		print("RESULT: PRESENT_IDLE")
		quit(0 if zero_ok else 2)
		return

	var image: Image = bridge.call("velocity_image")
	print("bridge: DLSS ran, frame_count=%d jitter=%s internal_size=%s" % [
			int(bridge.call("frame_count")), jitter, bridge.call("internal_size")])
	if image == null:
		print("velocity_image=null (the field is RID-invalid or not the expected format)")
		print("RESULT: RAN_NO_FIELD")
		quit(2)
		return
	print("velocity_image=%dx%d format=%d" % [image.get_width(), image.get_height(), image.get_format()])
	print("RESULT: RAN_WITH_FIELD")
	quit(0)
