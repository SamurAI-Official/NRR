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

# Sub-pixel jitter, as a Halton (2, 3) sequence in [-0.5, 0.5] pixels. A temporal upscaler is given frames
# whose low-resolution samples land on a different sub-pixel position every frame; that is what lets it
# resolve detail the low resolution alone cannot. The bases are named rather than inlined because the
# sequence has to be identical for every capture - an offset that cannot be reproduced cannot be recorded
# meaningfully, and the model is trained on the recorded values.
const JITTER_BASES := Vector2i(2, 3)

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
	# A scene laid out for *temporal* data, which the seven above are not: they were designed while the backdrop
	# was invisible to the motion pass (see the mirror in _run), so a fresh capture of them measures 79.6% sky
	# with the little geometry it has moving 0.85 px per frame at the input grid. This one puts the wall close,
	# the objects in front of it, and dollies the camera fast enough that the wall's own screen-space motion is a
	# few pixels per frame - the regime a temporal upscaler exists for. Measured with tools/history_reuse_probe.py
	# and tools/godot_capture's own jitter self-test; the probe's `--data` wants a packed dataset.
	"temporal": {
		# 0.05 units per frame against a wall 2.0 away is about 2.4 px per frame on a 128-wide input grid. Over
		# a 400-frame capture that is 20 units of travel: the camera approaches the wall without reaching it, and
		# the objects sit between the two so occlusion and disocclusion actually happen.
		"camera_velocity": Vector3(0.05, 0.02, 0.0),
		"checker": 6,
		"backdrop_size": Vector2(24.0, 14.0),
		"backdrop_z": -2.0,
		"objects": [
			{"shape": "sphere", "color": Color(0.85, 0.35, 0.25), "scale": 0.40,
				"orbit": 0.35, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.0, "spin": 0.0,
				"position": Vector3(-0.60, 0.10, -1.10)},
			{"shape": "box", "color": Color(0.25, 0.55, 0.90), "scale": 0.35,
				"orbit": 0.25, "axis": Vector3(0.0, 0.0, 1.0), "phase": 1.1, "spin": 0.9,
				"position": Vector3(0.55, -0.25, -1.45)},
			{"shape": "sphere", "color": Color(0.90, 0.85, 0.30), "scale": 0.22,
				"orbit": 0.45, "axis": Vector3(1.0, 0.0, 0.0), "phase": 2.3, "spin": 0.0,
				"position": Vector3(0.10, 0.35, -0.95)},
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
			{"shape": "torus", "color": Color(0.35, 0.85, 0.95), "scale": 1.50,
				"orbit": 0.60, "axis": Vector3(0.0, 1.0, 0.0), "phase": 0.8, "spin": 0.9,
				"position": Vector3(0.40, 0.20, -4.3)},
			{"shape": "sphere", "color": Color(0.95, 0.30, 0.30), "scale": 1.00,
				"orbit": 1.20, "axis": Vector3(1.0, 0.0, 0.0), "phase": 1.9, "spin": 0.0,
				"position": Vector3(-0.70, -0.25, -4.8)},
			{"shape": "box", "color": Color(0.25, 0.90, 0.55), "scale": 1.25,
				"orbit": 0.0, "axis": Vector3(0.0, 0.0, 1.0), "phase": 0.0, "spin": 2.6,
				"position": Vector3(0.85, -0.45, -5.4)},
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
# "halton" moves the low-resolution sample every frame; "none" holds it still, which is the ablation that
# shows how much of the temporal gain comes from the jitter and how much from the model simply seeing a real
# low-resolution render instead of a filtered downscale.
var _jitter_mode := "halton"
var _jitter_log: Array = []
# Freeze the scene: no camera travel and no animation, so consecutive frames sample the *same* scene at
# different sub-pixel positions. That is the only configuration in which integrating frames measures
# antialiasing - with camera motion, averaging frames blurs motion instead of averaging samples, and no
# AA conclusion follows. The existing captures all move (that is what motion vectors are for), which is
# why none of them could show whether the temporal path resolves anything.
var _static := false

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

	# The frame a temporal upscaler actually receives: a real raster at the low resolution, where geometry
	# edges are aliased, rather than a filtered downscale of a clean image (which carries no aliasing for a
	# temporal pass to resolve). It shares the colour world, so it draws the same scene - the same objects,
	# lit by the same light - and only the projection and the target size differ. That is the only pair of
	# differences the comparison is allowed to depend on.
	var lowres_size := maxi(8, _size / 2)
	var lowres_view := SubViewport.new()
	lowres_view.size = Vector2i(lowres_size, lowres_size)
	lowres_view.world_3d = color_world
	lowres_view.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	lowres_view.transparent_bg = false
	add_child(lowres_view)

	var lowres_camera := Camera3D.new()
	lowres_camera.near = 0.05
	lowres_camera.far = 100.0
	lowres_view.add_child(lowres_camera)
	lowres_camera.current = true
	var calibration := _calibrate_jitter(lowres_camera, color_camera, lowres_size, 0.05)
	if calibration.is_empty():
		print("RESULT: FAIL")
		get_tree().quit(1)
		return

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
	# The wall's size and distance are the scene's, with the originals as defaults: how far away it sits is what
	# sets how much screen-space motion a dolly produces, and the seven earlier scenes (8 units) and a temporal
	# capture (2) want very different answers. See the "temporal" scene.
	var backdrop_size: Vector2 = scene.get("backdrop_size", Vector2(24.0, 14.0))
	var backdrop_z: float = scene.get("backdrop_z", -8.0)
	var backdrop := MeshInstance3D.new()
	var backdrop_mesh := PlaneMesh.new()
	backdrop_mesh.size = backdrop_size
	backdrop.mesh = backdrop_mesh
	backdrop.transform = Transform3D(Basis().rotated(Vector3.RIGHT, -PI * 0.5), Vector3(0.0, 0.0, backdrop_z))
	var backdrop_material := StandardMaterial3D.new()
	backdrop_material.albedo_texture = checker
	backdrop_material.albedo_color = Color(0.9, 0.9, 0.95)
	backdrop_material.uv1_scale = Vector3(CHECKER_UV_SCALE, CHECKER_UV_SCALE, 1.0)
	backdrop.material_override = backdrop_material
	color_view.add_child(backdrop)

	var motion_shader: Shader = load(MOTION_SHADER)
	# The backdrop needs a mirror in the motion pass, and it was missing - which was a data bug, not a content
	# choice. The colour pass drew this static textured wall over most of the frame while the depth/motion pass
	# reported *no geometry* there at all, so the packer marked those pixels sky: depth 0, validity 0, nothing to
	# trust. A temporal model was being trained on frames whose temporal inputs contradicted their own colour,
	# and the tell was measurable - geometry 20.4% of pixels against a colour mean of 0.238 where an empty
	# environment would read ~0.04 (tools/history_reuse_probe.py on a fresh 4-frame capture, before and after).
	#
	# Its transform is its own rather than a scene object's, because _place_objects() builds orbit/spin
	# transforms and a wall's *orientation* is part of its definition here. Static does not mean motionless: a
	# wall the camera dollies past has screen-space motion like anything else, which is exactly the motion a
	# temporal model needs most - large, uniform, and over texture.
	#
	# Open, and measured: once a wall is in this pass, the scene's *objects* stop appearing in it. With the
	# wall, every pixel of the depth channel reads the wall's distance (2.000 at 2 units, 7.998 at 8) and none
	# reads an object's, while the same objects render correctly in the colour pass (1.3% of pixels by their
	# albedo) and rendered correctly in the motion pass before the wall was added to it. Two ordering
	# hypotheses were tested and rejected - render_priority -1 and +1 on the wall's material, both leaving the
	# depth uniform - so the next step is the shader's two position outputs (POSITION and VERTEX are both
	# written, and which one Godot's depth prepass uses decides this) rather than the render order. It matters
	# beyond this scene: the packer's occlusion test compares two depth fields, so a capture in which overlapping
	# geometry keeps only one layer would report disocclusion that never happened.
	var backdrop_mirror := MeshInstance3D.new()
	backdrop_mirror.mesh = backdrop.mesh
	backdrop_mirror.transform = backdrop.transform
	var backdrop_motion := ShaderMaterial.new()
	backdrop_motion.shader = motion_shader
	backdrop_motion.set_shader_parameter("depth_scale", DEPTH_SCALE)
	backdrop_motion.set_shader_parameter("motion_scale", MOTION_SCALE)
	backdrop_mirror.material_override = backdrop_motion
	motion_view.add_child(backdrop_mirror)
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
	if _static:
		# A static capture still jitters; it just does not travel. Camera velocity and object animation
		# are the two things that would make "average the frames" mean something other than "average the
		# samples", so both are frozen here rather than at the call site.
		camera_velocity = Vector3.ZERO
	# One previous MVP per object, so frame 0 legitimately reports zero motion: it has no history, which
	# is the same no-previous-frame case the runtime has on its first frame and after a scene cut.
	var previous_mvps: Array = []
	for index in movers.size():
		previous_mvps.append(null)
	# And one for the backdrop, which is not in `movers`; null on the first frame, so frame 0 reports zero
	# motion for it too.
	var previous_backdrop_mvp: Variant = null

	# Before a single frame is written: prove the low-resolution sample really moves, and by how much. The
	# offset is derived from the camera's own projection matrix, so the mapping is exact by construction; the
	# render check is what catches the other failure mode - the offset is set but never reaches the renderer -
	# which would produce a plausible-looking dataset with no jitter in it at all.
	if _jitter_mode != "none":
		var verified := await _verify_jitter(lowres_camera, calibration, lowres_view, movers)
		if not verified:
			print("RESULT: FAIL")
			get_tree().quit(1)
			return

	for frame in _frames:
		var time := 0.0 if _static else float(frame) / FPS
		var camera_transform := Transform3D(Basis(), camera_velocity * float(frame))
		color_camera.transform = camera_transform
		motion_camera.transform = camera_transform
		# The low-resolution camera sits at the same position as the others, so its frame differs from the
		# target only by resolution and by the sub-pixel shift applied to its own projection.
		lowres_camera.transform = camera_transform
		var jitter := _jitter_for(frame)
		_apply_jitter(lowres_camera, calibration, jitter)
		_jitter_log.append([jitter.x, jitter.y])
		var view_inverse := camera_transform.affine_inverse()
		var projection: Projection = motion_camera.get_camera_projection()
		var object_transforms := _place_objects(movers, time)

		for index in movers.size():
			var mover: Dictionary = movers[index]
			var mvp: Projection = projection * Projection(view_inverse * object_transforms[index])
			var material: ShaderMaterial = mover["material"]
			material.set_shader_parameter("cur_mvp", mvp)
			var previous: Variant = previous_mvps[index]
			material.set_shader_parameter("prev_mvp", mvp if previous == null else previous)
			previous_mvps[index] = mvp

		# The backdrop's own MVP, from the same view and projection the movers use. It has to be here rather
		# than in `movers` because its placement is its own transform (see above), and it has to be updated per
		# frame for the same reason the movers are: a static surface still moves on screen when the camera does.
		var backdrop_mvp: Projection = projection * Projection(view_inverse * backdrop.transform)
		backdrop_motion.set_shader_parameter("cur_mvp", backdrop_mvp)
		backdrop_motion.set_shader_parameter(
			"prev_mvp", backdrop_mvp if previous_backdrop_mvp == null else previous_backdrop_mvp)
		previous_backdrop_mvp = backdrop_mvp

		await RenderingServer.frame_post_draw
		var color_image := color_view.get_texture().get_image()
		var motion_image := motion_view.get_texture().get_image()
		var lowres_image := lowres_view.get_texture().get_image()
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
		var lowres_ok := lowres_image.save_png(
			ProjectSettings.globalize_path("%s/lowres_%04d.png" % [_out_dir, frame]))
		if color_ok != OK or motion_ok != OK or lowres_ok != OK:
			print("FAILURE: could not write frame %d (color=%d motion=%d lowres=%d)"
				% [frame, color_ok, motion_ok, lowres_ok])
			print("RESULT: FAIL")
			get_tree().quit(1)
			return
		if frame % 8 == 0 or frame == _frames - 1:
			print("  frame %d/%d written" % [frame, _frames - 1])

	_write_manifest(int(scene["checker"]), camera_velocity, int(calibration["lowres_size"]))
	print("RESULT: PASS (%d frames, scene=%s)" % [_frames, _scene_id])
	get_tree().quit(0)


func _write_manifest(checker: int, camera_velocity: Vector3, lowres_size: int) -> void:
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
		# The low-resolution pass. Its existence is what makes the dataset usable by a temporal upscaler, so
		# what it is (a render, not a downscale) and how it was sampled are recorded rather than implied.
		"lowres_size": lowres_size,
		"lowres_files": "lowres_%04d.png (RGB8: a real render at lowres_size, not a filtered downscale)",
		"lowres_reads": "colour, in [0,1], read the same way color_%04d.png is: pixel / 255",
		"jitter_mode": _jitter_mode,
		# Whether the scene was frozen. A downstream tool that integrates frames needs to know this:
		# averaging frames is only averaging samples when nothing moved.
		"static": _static,
		"jitter_bases": [JITTER_BASES.x, JITTER_BASES.y],
		# Per frame, in low-resolution pixels. +x is right and +y is down, and the value is the shift of the
		# sampled image relative to the un-jittered projection: a feature at pixel x in the un-jittered frame
		# appears at pixel x + jitter.x. Verified against a one-pixel render before any frame is written
		# (see _verify_jitter), because a sign error here would train and evaluate perfectly consistently and
		# wrongly. Frame 0 has no history, so its offset is irrelevant but still recorded.
		"jitter_pixels": _jitter_log,
		"jitter_decode": "jitter = jitter_pixels[frame]; the low-res sample was shifted by (jx, jy) pixels",
	}
	var path := ProjectSettings.globalize_path("%s/manifest.json" % _out_dir)
	var handle := FileAccess.open(path, FileAccess.WRITE)
	if handle == null:
		print("FAILURE: could not write %s" % path)
		return
	handle.store_string(JSON.stringify(manifest, "  ") + "\n")
	handle.close()
	print("  manifest: %s" % path)


func _place_objects(movers: Array, time: float) -> Array:
	# Transforms are built here rather than read back from the nodes, so the matrix handed to the shader and
	# the matrix Godot uses to draw are the same value by construction. The nodes are set as a side effect so
	# the self-test can freeze the scene without duplicating this maths.
	var transforms: Array = []
	for mover in movers:
		var entry: Dictionary = mover["entry"]
		var basis: Basis = Basis.from_euler(entry["axis"] * (float(entry["spin"]) * time))
		basis = basis.scaled(Vector3.ONE * float(entry["scale"]))
		var offset: Vector3 = entry["axis"] * (sin(float(entry["orbit"]) * time + float(entry["phase"])) * 1.1)
		var object_transform := Transform3D(basis, entry["position"] + offset)
		var color_node: MeshInstance3D = mover["color"]
		var motion_node: MeshInstance3D = mover["motion"]
		color_node.transform = object_transform
		motion_node.transform = object_transform
		transforms.append(object_transform)
	return transforms


func _halton(index: int, base: int) -> float:
	# The radical inverse. Integer arithmetic only, so the sequence is bit-identical wherever it is walked -
	# which matters because the offsets are recorded and replayed, not recomputed by the consumer.
	var result := 0.0
	var fraction := 1.0
	var value := index
	while value > 0:
		fraction /= float(base)
		result += fraction * float(value % base)
		value = floori(float(value) / float(base))
	return result


func _jitter_for(frame: int) -> Vector2:
	if _jitter_mode == "none":
		return Vector2.ZERO
	# frame + 1, because the radical inverse of 0 is 0, which would put the first frame exactly on the corner
	# of the [-0.5, 0.5] cell instead of inside it.
	var index := frame + 1
	return Vector2(_halton(index, JITTER_BASES.x) - 0.5, _halton(index, JITTER_BASES.y) - 0.5)


func _calibrate_jitter(camera: Camera3D, color_camera: Camera3D, lowres_size: int, near: float) -> Dictionary:
	# Two things are measured from the cameras' own projection matrices - the values the renderer itself uses
	# - rather than derived from a formula: how `size` sets the frustum, and how `frustum_offset` moves the
	# matrix. Godot's own temporal antialiasing shifts the projection by adding to columns[2][0] and [2][1],
	# and that term is added to NDC after the perspective divide and does not depend on the point, so the
	# image moves rigidly and one measured gain per axis is the complete relation. Nothing here can be wrong
	# in sign or scale without the self-test that follows failing.
	camera.projection = Camera3D.PROJECTION_FRUSTUM
	camera.frustum_offset = Vector2.ZERO
	camera.near = near
	camera.far = 100.0
	camera.size = 1.0
	var unit_height := 2.0 * near / camera.get_camera_projection().y.y
	camera.size = 2.0
	var double_height := 2.0 * near / camera.get_camera_projection().y.y
	if unit_height <= 0.0 or absf(double_height - 2.0 * unit_height) > 1e-4 * unit_height:
		print("FAILURE: frustum size does not scale the frustum height linearly (%.6f, %.6f)"
			% [unit_height, double_height])
		return {}

	# Match the perspective camera's framing exactly, so the only differences between the two views are the
	# resolution and the jitter - nothing else may vary, or the comparison measures the wrong thing.
	var wanted := 2.0 * near / color_camera.get_camera_projection().y.y
	camera.size = wanted / unit_height
	var base: Projection = camera.get_camera_projection()
	var height := 2.0 * near / base.y.y
	var width := 2.0 * near / base.x.x
	if absf(height - wanted) > 1e-3 * wanted:
		print("FAILURE: frustum camera frames %.6f, perspective camera frames %.6f" % [height, wanted])
		return {}

	camera.frustum_offset = Vector2(1.0, 0.0)
	var gain_x := camera.get_camera_projection().z.x - base.z.x
	camera.frustum_offset = Vector2(0.0, 1.0)
	var gain_y := camera.get_camera_projection().z.y - base.z.y
	camera.frustum_offset = Vector2.ZERO
	if absf(gain_x) < 1e-9 or absf(gain_y) < 1e-9:
		print("FAILURE: frustum_offset does not reach the projection matrix (gains %.9f, %.9f)"
			% [gain_x, gain_y])
		return {}
	print("  lowres: %dpx, frustum %.4f x %.4f at near, jitter gain (%.4f, %.4f), offset per pixel (%.6f, %.6f)"
		% [lowres_size, width, height, gain_x, gain_y,
		   2.0 / (float(lowres_size) * gain_x), 2.0 / (float(lowres_size) * gain_y)])
	return {"size": camera.size, "lowres_size": lowres_size, "near": near,
			"height": height, "width": width, "gain_x": gain_x, "gain_y": gain_y}


func _apply_jitter(camera: Camera3D, calibration: Dictionary, jitter: Vector2) -> void:
	# One low-resolution pixel spans 2/size of NDC, and the matrix's [2][0] entry shifts NDC by -offset *
	# gain. Solving that for the offset that produces the wanted pixel shift is exact rather than approximate.
	var size := float(calibration["lowres_size"])
	camera.frustum_offset = Vector2(
		-2.0 * jitter.x / (size * float(calibration["gain_x"])),
		2.0 * jitter.y / (size * float(calibration["gain_y"])))


func _verify_jitter(camera: Camera3D, calibration: Dictionary, view: SubViewport, movers: Array) -> bool:
	# Freeze the scene and render the low-resolution view with the sample moved by exactly one pixel on each
	# axis. A one-pixel shift of the projection must produce a frame that is the un-jittered frame translated
	# by exactly one pixel. This is the only place the claim is checked against a render rather than against
	# the matrix it was derived from, and it catches the failure the derivation cannot: an offset that is set
	# but never reaches the renderer, which would leave a dataset that is internally consistent and wrong.
	camera.transform = Transform3D()
	_place_objects(movers, 0.0)
	var frames: Array[Image] = []
	for offset: Vector2 in [Vector2.ZERO, Vector2(1.0, 0.0), Vector2(0.0, 1.0)]:
		_apply_jitter(camera, calibration, offset)
		# Two awaits, because the first frame after a setting may already have been drawn under the previous.
		await RenderingServer.frame_post_draw
		await RenderingServer.frame_post_draw
		frames.append(view.get_texture().get_image())
	_apply_jitter(camera, calibration, Vector2.ZERO)
	var reference := frames[0]
	if _image_std(reference) <= 1e-4:
		print("FAILURE: the low-resolution view rendered a flat frame - it is drawing nothing")
		return false
	var moved_x := _mean_abs_difference(reference, frames[1], Vector2i(-1, 0))
	var still_x := _mean_abs_difference(reference, frames[1], Vector2i.ZERO)
	var moved_y := _mean_abs_difference(reference, frames[2], Vector2i(0, -1))
	var still_y := _mean_abs_difference(reference, frames[2], Vector2i.ZERO)
	print("  jitter self-test: +1px x -> translated %.5f vs unmoved %.5f, +1px y -> translated %.5f vs unmoved %.5f"
		% [moved_x, still_x, moved_y, still_y])
	# Translated must be near zero and clearly better than unmoved. A blank frame cannot pass, because it was
	# already refused above; a wrong sign fails, because then the translated comparison is the worse one.
	var ok := moved_x <= 0.02 and moved_y <= 0.02 and still_x >= 4.0 * moved_x and still_y >= 4.0 * moved_y
	if not ok:
		print("FAILURE: a one-pixel jitter did not translate the low-resolution render by one pixel")
	return ok


func _mean_abs_difference(base: Image, moved: Image, lookup: Vector2i) -> float:
	# Mean |moved[y][x] - base[y + lookup.y][x + lookup.x]| over the overlap. `lookup` says where in `base`
	# the pixel at (x, y) of `moved` should have come from: a frame translated by +1 in x moved its content
	# right, so the pixel now at x holds what base held at x - 1. The border is skipped, because content
	# shifted off the edge has nothing left to be compared against.
	var total := 0.0
	var counted := 0
	for y in range(maxi(0, -lookup.y), mini(moved.get_height(), base.get_height() - lookup.y)):
		for x in range(maxi(0, -lookup.x), mini(moved.get_width(), base.get_width() - lookup.x)):
			var one := moved.get_pixel(x, y)
			var other := base.get_pixel(x + lookup.x, y + lookup.y)
			total += absf(one.r - other.r) + absf(one.g - other.g) + absf(one.b - other.b)
			counted += 3
	if counted == 0:
		return INF
	return total / float(counted)


func _image_std(image: Image) -> float:
	# Used only to refuse a blank low-resolution frame. If the shared World3D did not draw into the second
	# viewport, every comparison in the self-test would be zero and would pass; this makes that case loud.
	var total := 0.0
	var total_sq := 0.0
	var count := 0
	for y in image.get_height():
		for x in image.get_width():
			var pixel := image.get_pixel(x, y)
			var value := (pixel.r + pixel.g + pixel.b) / 3.0
			total += value
			total_sq += value * value
			count += 1
	if count == 0:
		return 0.0
	var mean := total / float(count)
	return sqrt(maxf(0.0, total_sq / float(count) - mean * mean))


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
			"--jitter":
				_jitter_mode = args[index + 1]
			"--static":
				_static = true
	if _jitter_mode != "halton" and _jitter_mode != "none":
		# Anything else would silently capture an un-jittered dataset labelled as jittered.
		print("FAILURE: unknown --jitter '%s' (expected 'halton' or 'none')" % _jitter_mode)
		_jitter_mode = "halton"


func _write_bytes(relative_path: String, data: PackedByteArray) -> int:
	# Raw bytes rather than an image format: the motion pass is half-float data, and every file format
	# Godot can write would quantise or reinterpret it. FAILED/OK are Godot's own error codes.
	var file := FileAccess.open(ProjectSettings.globalize_path(relative_path), FileAccess.WRITE)
	if file == null:
		return FAILED
	file.store_buffer(data)
	file.close()
	return OK

