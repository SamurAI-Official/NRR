extends SceneTree

# Temporal benchmark: what do NRR's temporal paths buy on a *jittered* moving sequence?
#
# Run headless:
#   & <godot_console> --headless --path engine_plugins/godot_verify --script res://benchmark_temporal.gd
#
# Why it is built this way:
#   * The sequence is synthesised from a high-resolution ground truth, so every frame has a *known* sub-pixel
#     sampling offset and a known translation. Fractional scene motion would have to be produced by
#     resampling the reference, and that blurs the reference - a confound large enough to swamp the effect
#     (it did, in tools/aa_resolve_probe.py, before it was found and fixed).
#   * Motion is whole ground-truth pixels, so each frame's reference is an exact shift of the ground truth:
#     edge and plain error then measure the reconstruction and not the harness.
#   * Metrics are the evaluation protocol's temporal pair (warping error, temporal PSNR) plus the fidelity
#     pair (edge-weighted and plain error against the frame's own reference).
#
# It also prints what the runtime *decided* per configuration, because "the pass ran", "the pass declined"
# and "the pass is off" are three different results and only one of them means the feature works.

const GROUND_W := 256
const GROUND_H := 192
const INPUT_W := 128
const INPUT_H := 96
const SCALE := 2
const FRAMES := 8
const MODEL := "res://models/nrr_upscaler_v0.1.onnx"
const OUTPUT_JSON := "user://nrr_temporal_benchmark.json"

## Whole-pixel translation per frame, in *ground-truth* pixels (so 2 = one render pixel per frame).
const SCENE_MOTION := 2


func _initialize() -> void:
	print("=== NRR temporal benchmark ===")
	print("godot=%s" % Engine.get_version_info().string)
	var nrr := NRR.new()
	if not nrr.initialize():
		print("RESULT: FAIL native binding unavailable: %s" % nrr.last_error)
		quit(1)
		return
	if not nrr.load_model(MODEL):
		print("RESULT: FAIL load_model: %s" % nrr.last_error)
		quit(1)
		return
	print("backend=%s model=%s" % [nrr.backend_name(), nrr.loaded_model])

	var ground := _ground_truth()

	# The configurations differ only in what the caller tells the runtime - same frames, same model, same
	# scene. That is the comparison that matters: the temporal path on or off.
	var runs := [
		{
			"name": "no-jitter",
			"note": "frames submitted with no offset and no measurement: what the addon did before",
			"jitter": false,
			"measure": false,
		},
		{
			"name": "jitter+magnitude",
			"note": "the offsets a temporal upscaler produces, and the measurement both bindings use",
			"jitter": true,
			"measure": true,
		},
	]

	var report := {"frames": FRAMES, "scene_motion_px": SCENE_MOTION, "runs": []}
	print("\n%-16s %-10s %-11s %-11s %-10s %-8s %s"
			% ["configuration", "edge", "plain", "warp-err", "t-psnr", "alpha", "runtime state"])
	for run in runs:
		var result := _run_configuration(nrr, ground, run)
		report["runs"].append(result)
		print("%-16s %-10.6f %-11.6f %-11.6f %-10.3f %-8.3f %s"
				% [run["name"], result["edge"], result["plain"], result["warping_error"],
				   result["temporal_psnr"], result["alpha"], result["phase"]])

	var file := FileAccess.open(OUTPUT_JSON, FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(report, "  "))
		file.close()
		print("\nwrote %s" % ProjectSettings.globalize_path(OUTPUT_JSON))

	var usable := false
	for run in report["runs"]:
		if str(run["phase"]).contains("phase-aligned") and not str(run["phase"]).contains("off"):
			usable = true
	print("integration_observed=%s" % usable)
	print("RESULT: %s" % ("PASS" if report["runs"].size() == runs.size() else "FAIL"))
	quit(0 if report["runs"].size() == runs.size() else 1)


func _run_configuration(nrr: NRR, ground: PackedFloat32Array, run: Dictionary) -> Dictionary:
	nrr.reset_temporal_history()
	var displayed: Array = []
	var sum_edge := 0.0
	var sum_plain := 0.0
	var last_state := {}
	for frame_index in range(FRAMES):
		# The frame's sampling offset: the Halton (2,3) sequence the training captures use. The scene moves
		# by whole ground-truth pixels, so the reference for the frame stays exact.
		var jitter: Vector2 = _halton(frame_index + 1) if run["jitter"] else Vector2.ZERO
		var offset := jitter + Vector2(SCENE_MOTION * frame_index, 0.0)
		var input := _render_low_res(ground, offset)
		if run["measure"]:
			# The caller's measurement of the scene's motion as a fraction of the frame width - what
			# NRR.measure_camera_motion() derives in a real scene, and exactly known here.
			nrr.set_motion_magnitude(float(SCENE_MOTION) / float(GROUND_W))
		nrr.set_jitter(jitter, run["jitter"])
		var image := nrr.render_frame(input, null, null)
		if _display_w == 0:
			# Discovered from the render rather than assumed - see the note above the metrics.
			_display_w = image.get_width()
			_display_h = image.get_height()
		var reference := _sample_grid(ground, _display_w, _display_h, SCENE_MOTION * frame_index)
		var weights := _edge_weights(reference)
		var as_float := _to_float(image)
		displayed.append(image)
		sum_edge += _edge_error(as_float, reference, weights)
		sum_plain += _plain_error(as_float, reference)
		last_state = nrr.temporal_state()
	var metrics := _temporal_metrics(displayed, maxi(SCENE_MOTION * _display_w / GROUND_W, 1))
	return {
		"name": run["name"],
		"note": run["note"],
		"display_size": [_display_w, _display_h],
		"edge": sum_edge / float(FRAMES),
		"plain": sum_plain / float(FRAMES),
		"warping_error": metrics["warping_error"],
		"temporal_psnr": metrics["temporal_psnr"],
		"alpha": float(last_state.get("temporal_alpha", 0.0)),
		"phase": str(last_state.get("debug_info", "")),
		"history_frames": int(last_state.get("history_frames", 0)),
	}


# --- fixture ------------------------------------------------------------------------------------

## A high-frequency, non-periodic pattern - a radial chirp plus a step edge - so a sub-pixel difference in
## sampling shows up as a difference in the reconstruction rather than as noise.
func _ground_truth() -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(GROUND_W * GROUND_H * 3)
	for y in range(GROUND_H):
		for x in range(GROUND_W):
			var fx := float(x) / float(GROUND_W)
			var fy := float(y) / float(GROUND_H)
			var radius := sqrt(pow(fx - 0.5, 2.0) + pow(fy - 0.5, 2.0))
			var chirp := 0.5 + 0.5 * sin(140.0 * radius * radius)
			var band := 1.0 if fx > 0.62 else 0.0
			var value := clampf(0.55 * chirp + 0.45 * band, 0.0, 1.0)
			var p := (y * GROUND_W + x) * 3
			out[p] = value
			out[p + 1] = value * 0.85
			out[p + 2] = 1.0 - value
	return out


func _shift_ground(ground: PackedFloat32Array, dx: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(ground.size())
	for y in range(GROUND_H):
		for x in range(GROUND_W):
			var sx := clampi(x - dx, 0, GROUND_W - 1)
			var p := (y * GROUND_W + x) * 3
			var q := (y * GROUND_W + sx) * 3
			for c in range(3):
				out[p + c] = ground[q + c]
	return out


## One low-resolution render: the ground truth sampled on a grid displaced by `offset`, bilinear, in the
## pixel-centre convention the trainer and the capture both use.
func _render_low_res(ground: PackedFloat32Array, offset: Vector2) -> Image:
	var out := Image.create(INPUT_W, INPUT_H, false, Image.FORMAT_RGBA8)
	for y in range(INPUT_H):
		for x in range(INPUT_W):
			var sample := Vector2((float(x) + 0.5 + offset.x) * SCALE - 0.5,
					(float(y) + 0.5 + offset.y) * SCALE - 0.5)
			var rgb := _sample_rgb(ground, sample.x, sample.y)
			out.set_pixel(x, y, Color(rgb.x, rgb.y, rgb.z, 1.0))
	return out


func _sample_rgb(data: PackedFloat32Array, x: float, y: float) -> Vector3:
	var cx := clampf(x, 0.0, float(GROUND_W) - 1.001)
	var cy := clampf(y, 0.0, float(GROUND_H) - 1.001)
	var x0 := int(floor(cx))
	var y0 := int(floor(cy))
	var fx := cx - float(x0)
	var fy := cy - float(y0)
	var result := Vector3.ZERO
	for c in range(3):
		var top: float = data[(y0 * GROUND_W + x0) * 3 + c] * (1.0 - fx) \
				+ data[(y0 * GROUND_W + x0 + 1) * 3 + c] * fx
		var bottom: float = data[((y0 + 1) * GROUND_W + x0) * 3 + c] * (1.0 - fx) \
				+ data[((y0 + 1) * GROUND_W + x0 + 1) * 3 + c] * fx
		result[c] = top * (1.0 - fy) + bottom * fy
	return result


func _halton(index: int) -> Vector2:
	return Vector2(_radical_inverse(index, 2) - 0.5, _radical_inverse(index, 3) - 0.5)


func _radical_inverse(index: int, radix: int) -> float:
	var result := 0.0
	var fraction := 1.0
	var value := index
	while value > 0:
		fraction /= float(radix)
		result += fraction * float(value % radix)
		value /= radix
	return result


# --- metrics ------------------------------------------------------------------------------------
#
# The displayed size is discovered from the first frame rather than assumed: the shipped fixture model is a
# passthrough, so its output is the input size, while a trained upscaler's is 2x. Every metric below is
# measured at whatever size the renderer produced, against the ground truth sampled on that same grid - which
# keeps the comparison like-with-like either way, and means a trained model plugged in later needs no change
# here.

var _display_w := 0
var _display_h := 0


## The ground truth sampled on a `width x height` grid displaced by `motion` ground-truth pixels: what the
## scene looks like on the grid this frame is displayed at. Whole-pixel motion at 1x is an exact shift.
func _sample_grid(ground: PackedFloat32Array, width: int, height: int,
		motion: int) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(width * height * 3)
	var sx := float(GROUND_W) / float(width)
	var sy := float(GROUND_H) / float(height)
	for y in range(height):
		for x in range(width):
			var sample := Vector2((float(x) + 0.5) * sx - 0.5 + float(motion) * sx,
					(float(y) + 0.5) * sy - 0.5)
			var rgb := _sample_rgb(ground, sample.x, sample.y)
			var p := (y * width + x) * 3
			out[p] = rgb.x
			out[p + 1] = rgb.y
			out[p + 2] = rgb.z
	return out


func _to_float(image: Image) -> PackedFloat32Array:
	# Read through the raw byte buffer: a get_pixel() call per pixel is the slowest thing GDScript does, and
	# this runs over every displayed pixel of every frame.
	var bytes := image.get_data()
	var out := PackedFloat32Array()
	out.resize(_display_w * _display_h * 3)
	var p := 0
	for i in range(_display_w * _display_h):
		out[p] = float(bytes[i * 4]) / 255.0
		out[p + 1] = float(bytes[i * 4 + 1]) / 255.0
		out[p + 2] = float(bytes[i * 4 + 2]) / 255.0
		p += 3
	return out


## Normalised gradient magnitude of the reference: antialiasing moves edge error, and a plain mean over a
## mostly flat frame hides the whole effect in the flat fraction.
func _edge_weights(reference: PackedFloat32Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(_display_w * _display_h)
	var total := 0.0
	for y in range(_display_h):
		for x in range(_display_w):
			var p := (y * _display_w + x) * 3
			var right := (y * _display_w + mini(x + 1, _display_w - 1)) * 3
			var below := (mini(y + 1, _display_h - 1) * _display_w + x) * 3
			var gx := absf(reference[right] - reference[p])
			var gy := absf(reference[below] - reference[p])
			var magnitude := sqrt(gx * gx + gy * gy)
			out[y * _display_w + x] = magnitude
			total += magnitude
	if total > 0.0:
		for i in range(out.size()):
			out[i] /= total
	return out


func _edge_error(candidate: PackedFloat32Array, reference: PackedFloat32Array,
		weights: PackedFloat32Array) -> float:
	var sum := 0.0
	for y in range(_display_h):
		for x in range(_display_w):
			var w := weights[y * _display_w + x]
			if w <= 0.0:
				continue
			var p := (y * _display_w + x) * 3
			sum += w * (absf(candidate[p] - reference[p])
					+ absf(candidate[p + 1] - reference[p + 1])
					+ absf(candidate[p + 2] - reference[p + 2])) / 3.0
	return sum


func _plain_error(candidate: PackedFloat32Array, reference: PackedFloat32Array) -> float:
	var sum := 0.0
	for i in range(candidate.size()):
		sum += absf(candidate[i] - reference[i])
	return sum / float(candidate.size())


## The evaluation protocol's temporal pair: the mean absolute difference between consecutive displayed
## frames after warping the older one by the known scene motion - the residual shimmer a temporal path is
## supposed to remove - and the temporal PSNR derived from it.
func _temporal_metrics(frames: Array, dx: int) -> Dictionary:
	if frames.size() < 2:
		return {"warping_error": 0.0, "temporal_psnr": 0.0}
	var cache: Array = []
	for frame in frames:
		cache.append(_to_float(frame))
	var sum := 0.0
	var count := 0
	for i in range(1, cache.size()):
		var previous: PackedFloat32Array = cache[i - 1]
		var current: PackedFloat32Array = cache[i]
		for y in range(_display_h):
			for x in range(maxi(dx, 0), _display_w):
				var p := (y * _display_w + x) * 3
				var q := (y * _display_w + x - dx) * 3
				for c in range(3):
					sum += absf(current[p + c] - previous[q + c])
					count += 1
	var warping := sum / float(maxi(count, 1))
	var psnr := 0.0
	if warping > 0.0:
		psnr = 20.0 * log(1.0 / warping) / log(10.0)
	return {"warping_error": warping, "temporal_psnr": psnr}
