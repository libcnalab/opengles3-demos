#version 300 es
// Key 5: the shadow map itself, drawn into a corner viewport. The depth texture is read through a
// PLAIN sampler2D. That is defined only when the unit's comparison mode is NONE, so unit 3 has a
// sampler object bound (COMPARE_MODE NONE, NEAREST) whose state supersedes the texture's own
// COMPARE_REF_TO_TEXTURE (ES 3.0 3.8.2). Near the sun = dark, far = bright.
precision highp float;
precision highp sampler2D;  // samplers default to lowp in GLSL ES 3.00

in vec2 v_uv;
uniform sampler2D u_depth;
uniform vec2 u_range;  // depths shown as black and white

out vec4 o_color;

void main()
{
    float d = texture(u_depth, v_uv).r;
    o_color = vec4(vec3(clamp((d - u_range.x) / (u_range.y - u_range.x), 0.0, 1.0)), 1.0);
}
