#version 300 es
// Flat color. Most stencil experiments draw with the color mask off and only care about the
// stencil buffer; the readout and the visual panels use the color.
precision mediump float;
uniform vec4 u_color;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_color;
}
