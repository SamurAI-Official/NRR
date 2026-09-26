// NRR blit shader.
//
// Draws the NRR output texture over the viewport. Written in plain Godot 4
// shader language with no renderer-specific builtins, so the same material
// works under forward_plus, mobile and gl_compatibility - the post-process
// node is a CanvasLayer overdraw, not a render-pipeline hook, which is what
// lets one addon cover every hardware pipeline NRR supports.
shader_type canvas_item;

uniform sampler2D nrr_output : filter_linear, repeat_disable;
uniform float blend_amount : hint_range(0.0, 1.0) = 1.0;

void fragment() {
	vec4 src = texture(TEXTURE, UV);
	vec4 nrr = texture(nrr_output, UV);
	COLOR = vec4(mix(src.rgb, nrr.rgb, blend_amount), src.a);
}
