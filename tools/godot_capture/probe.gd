extends Node

# Numerical probe for the capture path.
#
# A fronto-parallel plane viewed by a laterally translating camera has an exact closed-form motion field -
# every pixel shifts by the same UV amount - so one render checks three things at once: that the shader's
# matrix arithmetic is right, that the [0,1] remap survives the readback, and that a linear 16-bit target
# carries the value without an sRGB detour. If any of those were wrong, the measured shift would disagree
# with the analytic one by far more than a quantisation step, and the probe fails loudly instead of
# quietly producing a training set of nonsense.

const SHADER_PATH := "res://shaders/motion.gdshader"
const PLANE_DISTANCE := 4.0
const FOV_DEGREES := 60.0
const VIEWPORT_SIZE := Vector2i(64, 64)
# B channel: view distance mapped into [0,1]. R,G: motion remapped with this scale, because a half-float
# target near 0.5 resolves only ~2^-11 and the motions under test are ~2e-3 in UV units.
const DEPTH_SCALE := 0.05
const MOTION_SCALE := 8.0
const TOLERANCE := 3.0e-4

var _out_dir := "user://capture"
var _frames := 4
var _shift := 0.01


func _ready() -> void:
	_parse_args()
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(_out_dir))

	var viewport := SubViewport.new()
	viewport.size = VIEWPORT_SIZE
	viewport.use_hdr_2d = true          # linear float target: no sRGB encoding on the way out
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(viewport)

	var camera := Camera3D.new()
	camera.fov = FOV_DEGREES
	camera.near = 0.05
	camera.far = 100.0
	viewport.add_child(camera)
	camera.current = true

	var plane := MeshInstance3D.new()
	var mesh := PlaneMesh.new()
	mesh.size = Vector2(50.0, 50.0)
	plane.mesh = mesh
	plane.position = Vector3(0.0, 0.0, -PLANE_DISTANCE)
	# PlaneMesh's normal is +Y, and the camera looks down -Z, so without this the plane is edge-on and the
	# only thing captured is the background - which is exactly what the first probe run measured (the
	# sampled value was linear(0.3), Godot's default clear colour, not a motion vector).
	plane.rotate_x(-PI * 0.5)
	viewport.add_child(plane)

	var material := ShaderMaterial.new()
	material.shader = load(SHADER_PATH)
	plane.material_override = material
	material.set_shader_parameter("depth_scale", DEPTH_SCALE)
	material.set_shader_parameter("motion_scale", MOTION_SCALE)

	await get_tree().process_frame

	var previous_mvp: Projection = _mvp(camera, plane)
	var failures := 0
	var expected := -_shift / (2.0 * PLANE_DISTANCE * tan(deg_to_rad(FOV_DEGREES) * 0.5))
	for frame in _frames:
		camera.position = Vector3(_shift * float(frame), 0.0, 0.0)
		var current_mvp: Projection = _mvp(camera, plane)
		material.set_shader_parameter("cur_mvp", current_mvp)
		material.set_shader_parameter("prev_mvp", previous_mvp)
		await RenderingServer.frame_post_draw

		var image := viewport.get_texture().get_image()
		var path := ProjectSettings.globalize_path("%s/frame_%03d.png" % [_out_dir, frame])
		var saved := image.save_png(path)
		var pixel := image.get_pixel(VIEWPORT_SIZE.x / 2, VIEWPORT_SIZE.y / 2)
		var motion_x := (pixel.r - 0.5) * 2.0 / MOTION_SCALE
		var motion_y := (pixel.g - 0.5) * 2.0 / MOTION_SCALE
		# Frame 0 has no history - the previous matrices are this frame's - so the correct motion there is
		# exactly zero. That is the no-previous-frame case the runtime also has, and it is checked rather
		# than skipped, because a shader that invented a vector for it would train a model on fiction.
		var target := 0.0 if frame == 0 else expected
		var error := absf(motion_x - target)
		var ok := saved == OK and error < TOLERANCE and absf(motion_y) < TOLERANCE
		# No %e in GDScript's formatter: an unsupported conversion throws and takes the diagnosis with it,
		# which is how the first run of this probe managed to fail without saying why.
		print("frame %d: format=%d motion_uv=(%.6f, %.6f) expected=%.6f error=%.8f view_distance=%.4f saved_ok=%s"
			% [frame, image.get_format(), motion_x, motion_y, target, error,
				pixel.b / DEPTH_SCALE, saved == OK])
		if not ok:
			failures += 1
			print("  FAILED: error=%.8f (limit %.6f) motion_y=%.8f saved_ok=%s"
				% [error, TOLERANCE, motion_y, saved == OK])
		previous_mvp = current_mvp

	print("RESULT: %s (%d of %d frames failed)" % ["PASS" if failures == 0 else "FAIL", failures, _frames])
	get_tree().quit(0 if failures == 0 else 1)


func _mvp(camera: Camera3D, node: Node3D) -> Projection:
	# Exactly the matrices handed to the shader, so the check tests the pipeline rather than Godot's
	# internal conventions. GDScript has no Projection * Transform3D operator, so world-to-view and
	# view-to-clip are built separately and combined as Projections.
	var world_to_view: Transform3D = camera.global_transform.affine_inverse() * node.global_transform
	return camera.get_camera_projection() * Projection(world_to_view)


func _parse_args() -> void:
	var args := OS.get_cmdline_user_args()
	for index in range(args.size()):
		match args[index]:
			"--out":
				_out_dir = args[index + 1]
			"--frames":
				_frames = int(args[index + 1])
			"--shift":
				_shift = float(args[index + 1])
	print("probe: out=%s frames=%d shift=%.4f" % [_out_dir, _frames, _shift])
