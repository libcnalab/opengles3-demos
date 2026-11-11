#version 300 es
// Shows the shadow map as a topographic gray image: a ramp over the whole depth range plus a repeating
// sawtooth ("iso-depth bands") that makes small depth steps, such as the top of a box, visible.
//
// The depth texture is read through a PLAIN sampler2D, which is only defined when the texture unit's
// comparison mode is NONE; the texture itself is in COMPARE_REF_TO_TEXTURE mode, so unit 1 has a sampler
// object bound whose state supersedes it (ES 3.0 3.8.2). sampler2D defaults to lowp in GLSL ES 3.00,
// hence the explicit highp.
precision highp float;
precision highp sampler2D;

in vec2 v_uv;
uniform sampler2D u_depth;
uniform vec2 u_range;  // depth values mapped to black and white by the ramp
uniform float u_bands; // sawtooth periods over the depth range [0, 1]
uniform vec2 u_view;   // the inset shows s,t in [u_view.x, u_view.x + u_view.y] of the shadow map

out vec4 o_color;

void main()
{
    vec2 edge = min(v_uv, 1.0 - v_uv);
    if (min(edge.x, edge.y) < 0.02)
    {
        o_color = vec4(1.0, 0.85, 0.25, 1.0);  // frame
        return;
    }
    float d = texture(u_depth, u_view.x + v_uv * u_view.y).r;
    float ramp = clamp((d - u_range.x) / (u_range.y - u_range.x), 0.0, 1.0);
    float g = 0.25 * ramp + 0.75 * fract(d * u_bands);
    o_color = vec4(vec3(g), 1.0);
}
