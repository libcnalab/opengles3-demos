#version 300 es
// Flat Lambert shading with one directional light. u_lit = 0 outputs u_color unchanged (the
// translucent "culled" overlay and its outline). main.cpp evaluates the same formula on the CPU.
precision highp float;

in vec3 v_normal;

uniform vec4 u_color;
uniform vec3 u_light;  // normalized, pointing towards the light
uniform float u_lit;

out vec4 o_color;

void main()
{
    float shade = 0.35 + 0.65 * max(dot(normalize(v_normal), u_light), 0.0);
    o_color = vec4(u_color.rgb * mix(1.0, shade, u_lit), u_color.a);
}
