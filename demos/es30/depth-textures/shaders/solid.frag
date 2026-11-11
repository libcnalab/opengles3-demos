#version 300 es
// Constant color: stencil bit read-out passes (glColorMask selects one channel per bit) and the
// per-object stencil tint of the depth-stencil panels.
precision mediump float;

uniform vec4 u_color;
out vec4 o_color;

void main()
{
    o_color = u_color;
}
