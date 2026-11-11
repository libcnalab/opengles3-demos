#version 320 es
// Cell D: the object is sent ONCE in object space; the geometry shader places six copies.
precision highp float;

layout(location = 0) in vec3 a_position;

out VS_OUT
{
    vec3 world;  // object-space position (named "world" for the block layout shared with the other cells)
} vs_out;

void main()
{
    vs_out.world = a_position;
    gl_Position = vec4(a_position, 1.0);  // unused
}
