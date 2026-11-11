#version 310 es
// INTERFACE MISMATCH (precision). Locations and types agree with the vertex programs, but the inputs
// are mediump while the vertex outputs are highp. Between separable programs "all matched input and
// output variables have identical precision qualification" is required (ES 3.2 section 7.4.1,
// GLSL ES 3.10 section 9.2.2 table: precision "Yes"), so the pipeline is invalid.
precision mediump float;

layout(location = 1) uniform vec3 u_bg;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_tint;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(v_uv.x < 0.5 ? v_tint : u_bg, 1.0);
}
