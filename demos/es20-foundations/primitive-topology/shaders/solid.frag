#version 300 es
// Fragment shaders have no default float precision in GLSL ES; one must be declared.
precision mediump float;

uniform vec4 u_color;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_color;
}
