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
	print("caps.fp16_hardware=%s" % str(caps.get("fp16_hardware", -1)))

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

	# The phase-aligned switch, end to end through the loaded extension. Asserted as an invariant rather
	# than as success, because whether this host's backend has an accumulator is not something the driver
	# can know: the point is that "off" and "cannot" never get confused, and that a setting which is
	# accepted is one the accumulator actually holds.
	var off_ok := nrr.set_phase_aligned_accumulation(false)
	var off_state := nrr.phase_aligned_accumulation()
	print("phase_aligned_supported=%s state_after_off=%d" % [off_ok, off_state])
	if off_ok and off_state != 0:
		_fail("the switch was accepted but the accumulator does not report it off (state %d)" % off_state)
		return
	if not off_ok and off_state != -1:
		_fail("the switch was refused but the accumulator reports state %d instead of -1 (cannot)"
			% off_state)
		return

	# What that integration integrates. 0 (the displayed frames) is what the switch above has always
	# produced, so it has to be the default; 1 (the input renders) is the temporal upscale. Same invariant as
	# the switch: accepted means the accumulator holds it, refused means -1 rather than a source.
	var source_default := nrr.phase_aligned_source()
	print("phase_aligned_default_source=%d" % source_default)
	if source_default != 0 and source_default != -1:
		_fail("the default source must be 0 (displayed frames) or -1 (cannot ask); got %d" % source_default)
		return
	var source_ok := nrr.set_phase_aligned_source(1)
	var source_after := nrr.phase_aligned_source()
	print("phase_aligned_set_source=%s source_after_set=%d" % [source_ok, source_after])
	if source_ok and source_after != 1:
		_fail("the source was accepted but the accumulator reports %d" % source_after)
		return
	if not source_ok and source_after != -1:
		_fail("the source was refused but the accumulator reports %d instead of -1 (cannot)" % source_after)
		return
	nrr.set_phase_aligned_source(source_default)

	var on_ok := nrr.set_phase_aligned_accumulation(true)
	var on_state := nrr.phase_aligned_accumulation()
	print("phase_aligned_enabled=%s state_after_on=%d" % [on_ok, on_state])
	if on_ok and on_state != 1:
		_fail("the switch was accepted but the accumulator does not report it on (state %d)" % on_state)
		return
	if not on_ok and on_state != -1:
		_fail("enabling was refused but the accumulator reports state %d instead of -1 (cannot)"
			% on_state)
		return

	# A frame rendered with the switch on must still come back non-passthrough: this binding's frames
	# carry no jitter, so the runtime declines to integrate them - which is the right answer, and the one
	# that keeps a mean of identically-phased frames from being reported as antialiasing.
	if on_ok:
		var integrated := nrr.render_frame(source)
		if nrr.last_render_was_passthrough:
			_fail("the frame after enabling the integration came back as passthrough")
			return
		if integrated == null:
			_fail("the frame after enabling the integration was null")
			return
	nrr.set_phase_aligned_accumulation(false)
	print("phase_aligned_final_state=%d" % nrr.phase_aligned_accumulation())

	# The disocclusion guard, the same way and for the same reason: an invariant rather than a success,
	# because whether this host's backend accumulates is not something the driver can know, and because the
	# guard's default is the device's temporal-coherence capability rather than off - so the point is that
	# "off" and "cannot" never get confused, and that an accepted setting is one the accumulator holds.
	var guard_off_ok := nrr.set_disocclusion_rejection(false)
	var guard_off_state := nrr.disocclusion_rejection()
	print("disocclusion_supported=%s state_after_off=%d" % [guard_off_ok, guard_off_state])
	if guard_off_ok and guard_off_state != 0:
		_fail("the guard switch was accepted but the accumulator does not report it off (state %d)"
			% guard_off_state)
		return
	if not guard_off_ok and guard_off_state != -1:
		_fail("the guard switch was refused but the accumulator reports state %d instead of -1 (cannot)"
			% guard_off_state)
		return

	var guard_on_ok := nrr.set_disocclusion_rejection(true)
	var guard_on_state := nrr.disocclusion_rejection()
	print("disocclusion_enabled=%s state_after_on=%d" % [guard_on_ok, guard_on_state])
	if guard_on_ok and guard_on_state != 1:
		_fail("the guard switch was accepted but the accumulator does not report it on (state %d)"
			% guard_on_state)
		return
	if not guard_on_ok and guard_on_state != -1:
		_fail("enabling the guard was refused but the accumulator reports state %d instead of -1 (cannot)"
			% guard_on_state)
		return
	nrr.set_disocclusion_rejection(false)
	print("disocclusion_final_state=%d" % nrr.disocclusion_rejection())

	# --- The released scale-agnostic model, through the addon, at more than one tier ---------------------
	# This is the integration the addon README documents as "put it on a CanvasLayer and install the
	# released model", and two of its claims can only be checked from inside a running project: that
	# installing the addon and installing the model really is one act (NRRPostProcess.RELEASE_MODEL_PATH
	# resolves here), and that the released model renders a changed frame at a tier the caller named only by
	# the size of the image it handed over - the resolution token is derived inside the runtime.
	var post_process = load("res://addons/nrr/nrr_post_process.gd")
	var release_path: String = ""
	if post_process != null:
		release_path = post_process.RELEASE_MODEL_PATH
	print("upscaler_release_model_path=%s exists=%s"
		% [release_path, release_path != "" and FileAccess.file_exists(release_path)])
	if post_process == null:
		_fail("nrr_post_process.gd did not load from the addon")
		return
	# FileAccess rather than ResourceLoader: a .onnx is not a Godot resource type, so ResourceLoader.exists()
	# answers false for every model in the project, installed or not - which is exactly the kind of "it says
	# no" this driver exists to not fall for.
	if release_path == "" or not FileAccess.file_exists(release_path):
		_fail("the released model is not installed at '%s' - see the addon README's 'Using it as an upscaler'" % release_path)
		return
	if not _verify_upscaler(nrr, release_path):
		return

	nrr.shutdown()
	print("available_after_shutdown=%s" % nrr.available)
	print("RESULT: PASS")
	get_tree().quit(0)


# Renders the released model at three tiers through the addon. For each one: the model ran, the frame came back
# at the caller's size, and the pixels changed. The binding reports a passthrough by returning the caller's own
# image, so `last_render_was_passthrough` is exact rather than a guess, and a frame presented at the caller's
# size is the contract a CanvasLayer overdraw needs.
#
# The tiers are named by width - 128, 192, 256 - which is how a game gets one: the token that follows from each
# (0.0, 0.585, 1.0) is derived inside the runtime from the frame itself. 192 is between the two tiers this model
# was trained on, which is the case a two-tier model has to survive; tests/unit/test_scale_token.cpp is what
# reads the token's value back off a rendered frame, which GDScript cannot do.
func _verify_upscaler(nrr: NRR, model_path: String) -> bool:
	if not nrr.load_model(model_path):
		_fail("the released model did not load: %s" % nrr.last_error)
		return false
	print("upscaler_model=%s" % nrr.model_info())
	for size: Vector2i in [Vector2i(128, 96), Vector2i(192, 144), Vector2i(256, 192)]:
		var source := _gradient(size.x, size.y)
		var output: Image = nrr.render_frame(source)
		if output == null:
			_fail("the released model returned no frame at %dx%d" % [size.x, size.y])
			return false
		if nrr.last_render_was_passthrough:
			_fail("the released model passed the %dx%d frame through instead of rendering it: %s"
				% [size.x, size.y, nrr.last_error])
			return false
		if output.get_width() != size.x or output.get_height() != size.y:
			_fail("the released model returned a %dx%d frame for a %dx%d input, and a caller is presented with a frame at its own size"
				% [output.get_width(), output.get_height(), size.x, size.y])
			return false
		var delta := 0.0
		for y in size.y:
			for x in size.x:
				delta += absf(output.get_pixel(x, y).r - source.get_pixel(x, y).r)
		var mean := delta / float(size.x * size.y)
		print("upscaler_tier=%dx%d render_ms=%.3f mean_abs_dr=%.6f"
			% [size.x, size.y, nrr.last_render_time_ms(), mean])
		if mean <= 0.0:
			_fail("the %dx%d frame came back holding the input's own pixels" % [size.x, size.y])
			return false
	nrr.reset_temporal_history()
	return true


# The gradient the driver above builds, at any size rather than at one constant.
func _gradient(width: int, height: int) -> Image:
	var image := Image.create_empty(width, height, false, Image.FORMAT_RGBA8)
	for y in height:
		for x in width:
			var v := float(x + y) / float(width + height)
			image.set_pixel(x, y, Color(v, 1.0 - v, 0.5, 1.0))
	return image


func _fail(reason: String) -> void:
	print("FAILURE: %s" % reason)
	print("RESULT: FAIL")
	get_tree().quit(1)
