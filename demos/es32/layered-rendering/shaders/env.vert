#version 320 es
// Environment pass: the room (walls, decals, boxes) is sent ONCE as a triangle list with a flat colour per quad.
// The geometry shader renders it into all six faces of the cube map.
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

out VS_OUT
{
    vec3 color;
} vs_out;

void main()
{
    vs_out.color = a_color;
    gl_Position = vec4(a_position, 1.0);  // world space; the geometry shader applies the per-face matrix
}
