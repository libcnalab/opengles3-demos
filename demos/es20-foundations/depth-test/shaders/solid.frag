#version 300 es
// Flat color; the fragment depth is the fixed-function value interpolated from the vertices.
precision mediump float;
uniform vec4 u_color;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_color;
}
