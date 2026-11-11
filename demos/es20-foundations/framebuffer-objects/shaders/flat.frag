#version 300 es
// One uniform color: the depth and stencil scenes only care *where* and *whether* a fragment is
// written, so their rectangles are flat colored.
precision highp float;
uniform vec4 u_color;
out vec4 o_color;
void main()
{
    o_color = u_color;
}
