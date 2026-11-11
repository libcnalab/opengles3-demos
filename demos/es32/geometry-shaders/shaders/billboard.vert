#version 320 es
// Cell A: one vertex per particle. The vertex shader only moves the particle center into eye space;
// the geometry shader turns the point into a camera-facing quad.
precision highp float;

layout(location = 0) in vec3 a_position;  // particle center in world space
layout(location = 1) in float a_size;     // half extent of the sprite (eye-space units)

layout(location = 0) uniform mat4 u_view;

out VS_OUT
{
    float size;
} vs_out;

void main()
{
    gl_Position = u_view * vec4(a_position, 1.0);  // eye space; the geometry shader projects
    vs_out.size = a_size;
}
