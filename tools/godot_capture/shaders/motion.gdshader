shader_type spatial;

// Motion-vector and depth pass.
//
// The two matrices are applied to the model-space vertex directly, which is the same arithmetic a renderer
// performs when it writes a motion-vector buffer: the vector is derived from geometry and camera state,
// not estimated from pixels. That distinction is the whole point of capturing in the engine instead of
// computing optical flow from video.
//
// Output layout, and why it is three channels and not four: a Godot 2D HDR render target comes back as
// RGBH (format 14, no alpha), so validity cannot ride in A. Validating a vector - reprojected off-screen,
// or occluded last frame - is done in the packer instead, where two consecutive depth fields are
// available and the same arithmetic can run on the CPU. This shader therefore writes:
//   R,G = motion * motion_scale * 0.5 + 0.5      (so the half-float target resolves small motions)
//   B   = view distance scaled into [0,1] by depth_scale
//
// Motion is in viewport UV units with +y down. motion_scale exists because half floats carry ~11 bits of
// mantissa: without it, a 0.002 UV shift would land within a couple of quantisation steps of 0.5.

render_mode unshaded, skip_vertex_transform, cull_disabled, depth_draw_always;

uniform mat4 cur_mvp;
uniform mat4 prev_mvp;
uniform float depth_scale = 0.05;
uniform float motion_scale = 8.0;

varying vec4 cur_clip;
varying vec4 prev_clip;

void vertex() {
	// With skip_vertex_transform, VERTEX arrives in model space, so both transforms are explicit here.
	cur_clip = cur_mvp * vec4(VERTEX, 1.0);
	prev_clip = prev_mvp * vec4(VERTEX, 1.0);
	// One source of truth for the position: VERTEX in view space, which Godot then projects. Both POSITION and
	// VERTEX used to be written, and they are equal only while cur_mvp == PROJECTION_MATRIX * VIEW_MATRIX *
	// MODEL_MATRIX - true of the matrices this capture hands the shader, but the rasterizer uses one output and
	// the depth prepass the other, so any disagreement between them shows up as geometry that is drawn where
	// the depth buffer does not have it. That was measured: with a wall in the pass, every pixel of the depth
	// channel read the *wall's* distance and none read an object's, while the same objects rendered correctly
	// in colour. Writing one output makes the two agree by construction rather than by coincidence.
	VERTEX = (VIEW_MATRIX * MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
}

void fragment() {
	vec2 cur_uv = (cur_clip.xy / cur_clip.w) * 0.5 + 0.5;
	vec2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5 + 0.5;
	vec2 motion = cur_uv - prev_uv;
	ALBEDO = vec3(motion * motion_scale * 0.5 + 0.5,
		clamp(cur_clip.w * depth_scale, 0.0, 1.0));
}
