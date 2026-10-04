extends Node

# Stage 1b: the motion-grounded capture.
#
# Every frame is described entirely by its index - camera position and every object transform come from
# `frame`, never from delta time - so a capture is reproducible from (this revision, the scene id, the
# frame count) with no dependence on how fast the machine happened to render it.
#
# Two scene configurations exist, and the difference between them is the point: "held out" has to mean
# different content (layout, colours, texture frequency, camera path and motion magnitude), or the
# validation set measures memorisation of one shot rather than generalisation to another.
#
# Every object carries two things: a StandardMaterial3D for the colour pass, and its own instance of the
# motion shader for the motion/depth pass. The motion material has to be per object, because the matrix it
# is given is that object's MVP - a shared material could only hold one of them.

const MOTION_SHADER := "res://shaders/motion.gdshader"
const FPS := 30.0
const DEPTH_SCALE := 0.1
const MOTION_SCALE := 8.0

# Objects orbit in front of the camera at z < 0, over a static textured backdrop so the data also contains
# regions with exactly zero motion - which is what the runtime sees over most of a real frame.
const SCENES := {
	"train": {
		# Camera speed is per *frame* over a capture of hundreds of frames, so a value chosen by watching a
		# ten-frame run is wrong: 0.02 per frame carried the camera 8 units past the objects by frame 400 and
		# the data margin decayed from 0.020 to below the 0.0100 gate, which the packer then refused. This
		# keeps the whole capture inside 1.6 units, where the scene stays as designed.
		"camera_velocity": Vector3(0.004, 0.0, 0.0),
		"checker": 8,
		"objects": [
			{"shape": "sphere", "color": Color(0.85, 0.35, 0.25), "scale": 0.95,
				"orbit": 1.10, "axis": Vector3(1.0, 0.0, 0.0), "phase": 0.0, "spin": 0.0,
				"position": Vector3(0.0, 0.0, -2.8)},
			{"shape": "box", "color": Color(0.25, 0.55, 0.90), "scale": 1.05,
				"orbit": 0.0, "axis": Vector3(0.0, 0.0, 1.0), "phase": 0.0, "spin": 0.8,
				"position": Vector3(-1.10, 0.20, -3.4)},
			{"shape": "sphere", "color": Color(0.90, 0.85, 0.30), "scale": 0.55,
				"orbit": 0.65, "axis": Vector3(0.0, 0.0, 1.0), "phase": 1.7, "spin": 0.0,
				"position": Vector3(0.95, -0.30, -2.4)},
		],
	},
	"heldout": {
		# Objects sit further out and the dolly is halved, because the earlier version passed straight through
		# its own geometry: the check found surfaces 0.72 units away with 0.066 UV per frame of motion.
		"camera_velocity": Vector3(-0.006, 0.0, -0.004),
		"checker": 4,
		"objects": [
			{"shape": "torus", "color": Color(0.30, 0.80, 0.55), "scale": 1.60,
				"orbit": 0.35, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.4, "spin": 1.6,
				"position": Vector3(0.50, 0.0, -4.6)},
			{"shape": "box", "color": Color(0.95, 0.65, 0.20), "scale": 1.40,
				"orbit": 0.0, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.0, "spin": 2.4,
				"position": Vector3(-1.00, -0.35, -5.2)},
			{"shape": "sphere", "color": Color(0.80, 0.80, 0.95), "scale": 0.85,
				"orbit": 1.60, "axis": Vector3(1.0, 0.0, 0.0), "phase": 2.6, "spin": 0.0,
				"position": Vector3(0.0, 0.50, -5.8)},
		],
	},
	# More training content. Capacity is measured to be cheap in latency (eight times the parameters cost
	# 1.03-1.42x the time), so the frontier sweep will want a model with more room - and a model with more
	# room needs more than 367 pairs. These are separate scenes rather than a longer capture of the same one,
	# because camera speed is per frame and a long capture drifts out of the useful range, which the data
	# gate then refuses.
	"train2": {
		"camera_velocity": Vector3(0.005, 0.002, 0.0),
		"checker": 3,
		"objects": [
			{"shape": "box", "color": Color(0.70, 0.25, 0.65), "scale": 1.10,
				"orbit": 0.90, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.9, "spin": 0.6,
				"position": Vector3(-0.30, 0.10, -2.9)},
			{"shape": "sphere", "color": Color(0.20, 0.75, 0.80), "scale": 0.70,
				"orbit": 0.50, "axis": Vector3(1.0, 0.0, 0.0), "phase": 2.1, "spin": 0.0,
				"position": Vector3(0.90, -0.10, -3.3)},
			{"shape": "torus", "color": Color(0.90, 0.85, 0.35), "scale": 0.55,
				"orbit": 1.30, "axis": Vector3(0.0, 0.0, 1.0), "phase": 1.3, "spin": 1.9,
				"position": Vector3(-1.00, -0.40, -3.7)},
		],
	},
	"train3": {
		# Finer texture and a different palette, so the training set is not one look: the procedural pairs
		# failed their detail gate for exactly this reason once, having almost no high-frequency content.
		"camera_velocity": Vector3(-0.004, 0.0, 0.003),
		"checker": 16,
		"objects": [
			{"shape": "torus", "color": Color(0.95, 0.45, 0.60), "scale": 1.30,
				"orbit": 0.70, "axis": Vector3(1.0, 0.0, 0.0), "phase": 0.2, "spin": 0.0,
				"position": Vector3(0.20, 0.20, -3.1)},
			{"shape": "sphere", "color": Color(0.35, 0.40, 0.95), "scale": 0.60,
				"orbit": 1.10, "axis": Vector3(0.0, 1.0, 0.0), "phase": 2.8, "spin": 1.2,
				"position": Vector3(-0.85, 0.05, -2.6)},
			{"shape": "box", "color": Color(0.55, 0.90, 0.45), "scale": 0.80,
				"orbit": 0.0, "axis": Vector3(0.0, 0.0, 1.0), "phase": 0.0, "spin": 2.9,
				"position": Vector3(0.75, -0.35, -3.9)},
		],
	},
	"heldout2": {
		# A second held-out scene, so validation is not a single shot's worth of content.
		"camera_velocity": Vector3(0.004, -0.003, -0.003),
		"checker": 6,
		"objects": [
			{"shape": "sphere", "color": Color(0.90, 0.75, 0.25), "scale": 1.20,
				"orbit": 0.80, "axis": Vector3(0.0, 1.0, 0.0), "phase": 1.1, "spin": 2.1,
				"position": Vector3(-0.40, 0.15, -4.4)},
			{"shape": "box", "color": Color(0.45, 0.50, 0.85), "scale": 1.00,
				"orbit": 0.45, "axis": Vector3(1.0, 0.0, 0.0), "phase": 0.6, "spin": 0.0,
				"position": Vector3(0.95, -0.20, -4.9)},
			{"shape": "torus", "color": Color(0.80, 0.35, 0.35), "scale": 0.90,
				"orbit": 1.40, "axis": Vector3(0.0, 0.0, 1.0), "phase": 2.3, "spin": 1.5,
				"position": Vector3(0.10, 0.55, -5.5)},
		],
	},
	# More held-out variety for the commercial comparison: a third and fourth validation scene, so "held
	# out" is not two shots. heldout3 uses a finer texture and a cool/vibrant palette; heldout4 uses a
	# coarser texture and a warm/earth palette. Both keep the objects past z = -4.4 and a dolly whose whole
	# 400-frame span stays well short of the geometry, for the same reason the first held-out did.
	"heldout3": {
		"camera_velocity": Vector3(0.005, -0.004, -0.002),
		"checker": 5,
		"objects": [
			{"shape": "torus", "color": Color(0.35, 0.85, 0.95), "scale": 1.10,
				"orbit": 0.60, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.8, "spin": 0.9,
				"position": Vector3(0.40, 0.20, -4.5)},
			{"shape": "sphere", "color": Color(0.95, 0.30, 0.30), "scale": 0.75,
				"orbit": 1.20, "axis": Vector3(1.0, 0.0, 0.0), "phase": 1.9, "spin": 0.0,
				"position": Vector3(-0.70, -0.25, -5.0)},
			{"shape": "box", "color": Color(0.25, 0.90, 0.55), "scale": 0.90,
				"orbit": 0.0, "axis": Vector3(0.0, 0.0, 1.0), "phase": 0.0, "spin": 2.6,
				"position": Vector3(0.85, -0.45, -5.6)},
		],
	},
	"heldout4": {
		"camera_velocity": Vector3(-0.004, 0.003, -0.003),
		"checker": 7,
		"objects": [
			{"shape": "sphere", "color": Color(0.95, 0.95, 0.90), "scale": 1.35,
				"orbit": 0.45, "axis": Vector3(0.0, 0.0, 1.0), "phase": 2.4, "spin": 1.4,
				"position": Vector3(-0.50, 0.30, -4.7)},
			{"shape": "box", "color": Color(0.55, 0.35, 0.25), "scale": 1.15,
				"orbit": 0.90, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.3, "spin": 0.0,
				"position": Vector3(0.70, -0.15, -5.2)},
			{"shape": "torus", "color": Color(0.40, 0.55, 0.75), "scale": 0.65,
				"orbit": 1.50, "axis": Vector3(1.0, 0.0, 0.0), "phase": 3.5, "spin": 1.8,
				"position": Vector3(0.05, -0.55, -5.9)},
		],
	},
}

var _scene_id := "train"
var _frames := 24
var _size := 512
var _out_dir := "user://capture"

# How many times the checker tiles across a unit of surface. This exists because the first capture failed
# the data gate for a reason worth writing down: a 64-pixel texture stretched over a 24-unit backdrop puts
# each square about a fifth of the frame wide, which is not high-frequency content at all, and the pair
# margin came out at 0.0063 against a 0.0100 bar. The gate refused it, correctly.
const CHECKER_UV_SCALE := 12.0


func _ready() -> void:
	_parse_args()
	# An unknown scene id must stop here: quitting from a helper defers the quit, and _run() would still
	# execute and write a dataset for a fallback scene the caller never asked for.
	if not SCENES.has(_scene_id):
		print("FAILURE: unknown scene '%s' (known: %s)" % [_scene_id, SCENES.keys()])
		print("RESULT: FAIL")
		get_tree().quit(1)
		return
	print("capture: scene=%s frames=%d size=%d out=%s" % [_scene_id, _frames, _size, _out_dir])
	_run()


func _run() -> void:
	var scene: Dictionary = SCENES[_scene_id]
	# Two worlds rather than one. The colour pass needs a lit environment; the motion pass needs a
	# background that cannot be mistaken for data. Sharing a world was the first attempt, and it meant a
	# pixel with no geometry carried the environment colour - (0.0272, 0.0331, 0.0550), which is linear of
	# the sky colour - and that decodes as a perfectly plausible motion vector of -0.118 and a distance of
	# 0.55. With its own world clearing to black, "no geometry" means raw zero and nothing else.
	var color_world := World3D.new()
	var motion_world := World3D.new()

	var color_view := SubViewport.new()
	color_view.size = Vector2i(_size, _size)
	color_view.world_3d = color_world
	color_view.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	color_view.transparent_bg = false
	add_child(color_view)

	var motion_view := SubViewport.new()
	motion_view.size = Vector2i(_size, _size)
	motion_view.use_hdr_2d = true          # linear float target: the vectors are not sRGB-encoded
	motion_view.world_3d = motion_world
	motion_view.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	motion_view.transparent_bg = false
	add_child(motion_view)

	var color_camera := Camera3D.new()
	color_camera.fov = 60.0
	color_camera.near = 0.05
	color_camera.far = 100.0
	color_view.add_child(color_camera)
	color_camera.current = true

	var motion_camera := Camera3D.new()
	motion_camera.fov = 60.0
	motion_camera.near = 0.05
	motion_camera.far = 100.0
	motion_view.add_child(motion_camera)
	# Without this the motion viewport renders nothing: a viewport with no current camera draws an empty
	# target, which read back as all-zero - decoded as a uniform "motion" of -0.125 in both channels, and
	# the check script caught it as a failed capture rather than as training data.
	motion_camera.current = true

	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.18, 0.20, 0.26)
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color(0.35, 0.38, 0.45)
	environment.ambient_light_energy = 0.6
	color_world.environment = environment

	# Black, unlit, no ambient: anything the motion pass did not draw stays exactly zero.
	var empty := Environment.new()
	empty.background_mode = Environment.BG_COLOR
	empty.background_color = Color(0.0, 0.0, 0.0)
	empty.ambient_light_source = Environment.AMBIENT_SOURCE_DISABLED
	motion_world.environment = empty

	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-35.0, 25.0, 0.0)
	light.light_energy = 1.2
	color_view.add_child(light)

	var checker := _checker_texture(int(scene["checker"]))
	var backdrop := MeshInstance3D.new()
	var backdrop_mesh := PlaneMesh.new()
	backdrop_mesh.size = Vector2(24.0, 14.0)
	backdrop.mesh = backdrop_mesh
	backdrop.transform = Transform3D(Basis().rotated(Vector3.RIGHT, -PI * 0.5), Vector3(0.0, 0.0, -8.0))
	var backdrop_material := StandardMaterial3D.new()
	backdrop_material.albedo_texture = checker
	backdrop_material.albedo_color = Color(0.9, 0.9, 0.95)
	backdrop_material.uv1_scale = Vector3(CHECKER_UV_SCALE, CHECKER_UV_SCALE, 1.0)
	backdrop.material_override = backdrop_material
	color_view.add_child(backdrop)

	var motion_shader: Shader = load(MOTION_SHADER)
	var movers: Array[Dictionary] = []
	for entry in scene["objects"]:
		var node := MeshInstance3D.new()
		node.mesh = _mesh_for(String(entry["shape"]))
		var material := StandardMaterial3D.new()
		material.albedo_color = entry["color"]
		material.albedo_texture = checker
		material.uv1_scale = Vector3(CHECKER_UV_SCALE, CHECKER_UV_SCALE, 1.0)
		node.material_override = material
		color_view.add_child(node)

		var mirror := MeshInstance3D.new()
		mirror.mesh = node.mesh
		var motion_material := ShaderMaterial.new()
		motion_material.shader = motion_shader
		motion_material.set_shader_parameter("depth_scale", DEPTH_SCALE)
		motion_material.set_shader_parameter("motion_scale", MOTION_SCALE)
		mirror.material_override = motion_material
		motion_view.add_child(mirror)

		movers.append({"entry": entry, "color": node, "motion": mirror, "material": motion_material})

	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_out_dir))
	var camera_velocity: Vector3 = scene["camera_velocity"]
	# One previous MVP per object, so frame 0 legitimately reports zero motion: it has no history, which
	# is the same no-previous-frame case the runtime has on its first frame and after a scene cut.
	var previous_mvps: Array = []
	for index in movers.size():
		previous_mvps.append(null)

	for frame in _frames:
		var time := float(frame) / FPS
		var camera_transform := Transform3D(Basis(), camera_velocity * float(frame))
		color_camera.transform = camera_transform
		motion_camera.transform = camera_transform
		var view_inverse := camera_transform.affine_inverse()
		var projection: Projection = motion_camera.get_camera_projection()

		for index in movers.size():
			var mover: Dictionary = movers[index]
			var entry: Dictionary = mover["entry"]
			# Transforms are built here rather than read back from the nodes, so the matrix handed to the
			# shader and the matrix Godot uses to draw are the same value by construction.
			var basis: Basis = Basis.from_euler(entry["axis"] * (float(entry["spin"]) * time))
			basis = basis.scaled(Vector3.ONE * float(entry["scale"]))
			var offset: Vector3 = entry["axis"] * (sin(float(entry["orbit"]) * time + float(entry["phase"])) * 1.1)
			var object_transform := Transform3D(basis, entry["position"] + offset)
			var color_node: MeshInstance3D = mover["color"]
			var motion_node: MeshInstance3D = mover["motion"]
			color_node.transform = object_transform
			motion_node.transform = object_transform

			var mvp: Projection = projection * Projection(view_inverse * object_transform)
			var material: ShaderMaterial = mover["material"]
			material.set_shader_parameter("cur_mvp", mvp)
			var previous: Variant = previous_mvps[index]
			material.set_shader_parameter("prev_mvp", mvp if previous == null else previous)
			previous_mvps[index] = mvp

		await RenderingServer.frame_post_draw
		var color_image := color_view.get_texture().get_image()
		var motion_image := motion_view.get_texture().get_image()
		# The motion pass is written as raw half-float bytes rather than PNG: Godot's PNG writer converts a
		# half-float image down to 8 bits, which silently turned a 2e-3 vector into a 4e-3 quantisation step
		# and made the whole dataset nonsense. get_data() is the image's own bytes, so nothing is lost, and
		# the format is asserted rather than assumed.
		if motion_image.get_format() != Image.FORMAT_RGBH:
			print("FAILURE: motion target came back as format %d, expected RGBH (%d)"
				% [motion_image.get_format(), Image.FORMAT_RGBH])
			print("RESULT: FAIL")
			get_tree().quit(1)
			return
		var color_ok := color_image.save_png(
			ProjectSettings.globalize_path("%s/color_%04d.png" % [_out_dir, frame]))
		var motion_ok := _write_bytes("%s/motion_%04d.bin" % [_out_dir, frame], motion_image.get_data())
		if color_ok != OK or motion_ok != OK:
			print("FAILURE: could not write frame %d (color=%d motion=%d)" % [frame, color_ok, motion_ok])
			print("RESULT: FAIL")
			get_tree().quit(1)
			return
		if frame % 8 == 0 or frame == _frames - 1:
			print("  frame %d/%d written" % [frame, _frames - 1])

	_write_manifest(int(scene["checker"]), camera_velocity)
	print("RESULT: PASS (%d frames, scene=%s)" % [_frames, _scene_id])
	get_tree().quit(0)


func _write_manifest(checker: int, camera_velocity: Vector3) -> void:
	# Everything needed to interpret the data without guessing: the scales the shader used, the convention
	# the vectors follow, and the parameters that generated each frame. The packer adds file hashes.
	var manifest := {
		"scene": _scene_id,
		"frames": _frames,
		"size": _size,
		"fps": FPS,
		"motion_scale": MOTION_SCALE,
		"depth_scale": DEPTH_SCALE,
		"checker_period": checker,
		"camera_velocity": [camera_velocity.x, camera_velocity.y, camera_velocity.z],
		"motion_convention": "viewport UV units, +y down, from the current frame to the previous one",
		"motion_decode": "motion = (png_value - 0.5) * 2 / motion_scale",
		"depth_decode": "view_distance = (png_value_blue) / depth_scale",
		"color_files": "color_%04d.png (RGB8)",
		"motion_files": "motion_%04d.bin (RGB half-float, 6 bytes per pixel, row-major)",
	}
	var path := ProjectSettings.globalize_path("%s/manifest.json" % _out_dir)
	var handle := FileAccess.open(path, FileAccess.WRITE)
	if handle == null:
		print("FAILURE: could not write %s" % path)
		return
	handle.store_string(JSON.stringify(manifest, "  ") + "\n")
	handle.close()
	print("  manifest: %s" % path)


func _mesh_for(shape: String) -> Mesh:
	match shape:
		"box":
			var box := BoxMesh.new()
			box.size = Vector3.ONE
			return box
		"torus":
			var torus := TorusMesh.new()
			torus.inner_radius = 0.28
			torus.outer_radius = 0.62
			return torus
		_:
			var sphere := SphereMesh.new()
			sphere.radius = 0.5
			sphere.height = 1.0
			return sphere


func _checker_texture(period: int) -> ImageTexture:
	# A deterministic procedural texture: a checkerboard gives the pair real high-frequency content, which
	# is what a 2x upscaler is judged on, and it costs no committed asset.
	var image := Image.create(period * 8, period * 8, false, Image.FORMAT_RGB8)
	for y in image.get_height():
		for x in image.get_width():
			var on := ((x / period) + (y / period)) % 2 == 0
			image.set_pixel(x, y, Color(0.86, 0.86, 0.88) if on else Color(0.42, 0.44, 0.5))
	return ImageTexture.create_from_image(image)


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	for index in range(args.size()):
		match args[index]:
			"--scene":
				_scene_id = args[index + 1]
			"--frames":
				_frames = int(args[index + 1])
			"--size":
				_size = int(args[index + 1])
			"--out":
				_out_dir = args[index + 1]


func _write_bytes(relative_path: String, data: PackedByteArray) -> int:
	# Raw bytes rather than an image format: the motion pass is half-float data, and every file format
	# Godot can write would quantise or reinterpret it. FAILED/OK are Godot's own error codes.
	var file := FileAccess.open(ProjectSettings.globalize_path(relative_path), FileAccess.WRITE)
	if file == null:
		return FAILED
	file.store_buffer(data)
	file.close()
	return OK

