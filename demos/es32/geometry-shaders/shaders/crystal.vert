#version 320 es
// Cell B: the vertex shader passes the world-space position on; the geometry shader does the work.
precision highp float;

layout(location = 0) in vec3 a_position;

out VS_OUT
{
    vec3 world;
} vs_out;

void main()
{
    vs_out.world = a_position;
    gl_Position = vec4(a_position, 1.0);  // unused by the geometry shader
}
