#version 320 es
// Additive light halo: u_color * (1 - r^2)^2 inside the unit disc. Drawn with glBlendFunc(ONE, ONE) into the
// RGBA16F target, so the halos ADD to the scene values beyond 1.0 (an RGBA8 target would saturate).
precision highp float;

in vec2 v_uv;
uniform vec3 u_color;

layout(location = 0) out vec4 o_color;

void main()
{
    float f = max(1.0 - dot(v_uv, v_uv), 0.0);
    o_color = vec4(u_color * (f * f), 0.0);
}
