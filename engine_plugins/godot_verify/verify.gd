extends Node

# Verification driver for the NRR Godot addon. Runs headless and prints one
# unambiguous RESULT line, so a missing/not-loading extension cannot pass
# silently.

const W := 64
const H := 48
const WARMUP_FRAMES := 3
const MODEL := "res://models/nrr_upscaler_v0.1.onnx"


func _ready() -> void:
	print("=== NRR GDExtension verification ===")
	print("godot_version=%s" % Engine.get_version_info().string)

	var registered := ClassDB.class_exists("NRRNative")
	print("class_registered=%s" % registered)
	if not registered:
		_fail("NRRNative was not registered: the GDExtension did not load")
		return

	var nrr := NRR.new()
	print("binding_present=%s" % nrr.is_binding_present())
	print("library_version=%s" % nrr.library_version())
	print("entry_point_count_before_initialize=%d" % nrr.native_entry_point_count())
	print("available_before_initialize=%s" % nrr.available)

	if not nrr.initialize():
		_fail("initialize() failed: %s" % nrr.last_error)
		return
	print("available=%s" % nrr.available)
	print("backend=%s" % nrr.backend_name())
	print("entry_point_count=%d" % nrr.native_entry_point_count())
	# Capability reporting must be honest: before a session exists nothing has been
	# measured, so nothing may be claimed. Both sides are printed so a regression
	# that reintroduces an unmeasured claim shows up as a changed line here.
	var caps_before := nrr.capabilities()
	print("caps_before_load.active_backend=%s" % str(caps_before.get("active_backend", "")))
	print("caps_before_load.neural_acceleration=%s" % str(caps_before.get("neural_acceleration", -1)))
	print("caps_before_load.fp16=%s" % str(caps_before.get("fp16", -1)))

	var loaded := nrr.load_model(MODEL)
	print("load_model=%s path=%s" % [loaded, nrr.loaded_model])
	if not loaded:
		_fail("load_model failed: %s" % nrr.last_error)
		return
	print("model_info=%s" % nrr.model_info())

	# A session exists now, so the backend and the provider are measurable.
	var caps := nrr.capabilities()
	print("caps.active_backend=%s" % str(caps.get("active_backend", "")))
	print("caps.neural_acceleration=%s" % str(caps.get("neural_acceleration", -1)))
	print("caps.fp16=%s" % str(caps.get("fp16", -1)))

	var source := Image.create_empty(W, H, false, Image.FORMAT_RGBA8)
	for y in H:
		for x in W:
			var v := float(x + y) / float(W + H)
			source.set_pixel(x, y, Color(v, 1.0 - v, 0.5, 1.0))

	# The first frame carries one-time neural-stack initialization (CUDA/cuDNN
	# context creation, kernel autotuning, session warm-up). Reporting that one
	# number as "the render cost" is how a 632 ms cold frame was mistaken for the
	# frame budget. Measure the cold frame AND the steady state; assert the latter.
	var output := nrr.render_frame(source)
	var is_passthrough := nrr.last_render_was_passthrough
	print("render_out=%dx%d format=%d" % [output.get_width(), output.get_height(), output.get_format()])
	print("last_error=%s" % nrr.last_error)
	var first_frame_ms := nrr.last_render_time_ms()

	if is_passthrough:
		_fail("render_frame() passed the input through instead of rendering")
		return

	var delta := 0.0
	for y in H:
		for x in W:
			delta += absf(output.get_pixel(x, y).r - source.get_pixel(x, y).r)
	print("mean_abs_dr_vs_input=%.6f" % (delta / float(W * H)))

	# Warm up at the size we actually render at, then measure the steady state.
	var warmed := nrr.warmup(WARMUP_FRAMES, W, H)
	print("warmup=%s" % warmed)
	if not warmed:
		_fail("warmup() did not produce a real render")
		return

	var steady_frames := 5
	var steady_total := 0.0
	for i in steady_frames:
		nrr.render_frame(source)
		if nrr.last_render_was_passthrough:
			_fail("render became passthrough during the steady-state loop")
			return
		steady_total += nrr.last_render_time_ms()
	var steady_state_ms := steady_total / float(steady_frames)

	print("first_frame_ms=%.3f" % first_frame_ms)
	print("steady_state_ms=%.3f" % steady_state_ms)
	if steady_state_ms <= 0.0:
		_fail("steady-state render time was not measured")
		return
	# Generous factor on purpose: this catches a warm-up that never converges, and
	# is not meant to fail on millisecond noise in shared CI.
	if steady_state_ms > first_frame_ms * 2.0 + 1.0:
		_fail("steady state (%.3f ms) is far above the first frame (%.3f ms): warm-up is not converging"
			% [steady_state_ms, first_frame_ms])
		return
	print("frame_budget_ms=%.3f (steady state; %.3f ms cold)"
		% [steady_state_ms, first_frame_ms])

	print("reset_temporal_history=%s" % nrr.reset_temporal_history())

	nrr.shutdown()
	print("available_after_shutdown=%s" % nrr.available)
	print("RESULT: PASS")
	get_tree().quit(0)


func _fail(reason: String) -> void:
	print("FAILURE: %s" % reason)
	print("RESULT: FAIL")
	get_tree().quit(1)
