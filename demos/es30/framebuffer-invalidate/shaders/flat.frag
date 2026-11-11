#version 300 es
// One uniform color. The depth/stencil probe passes only care where a fragment is written.
precision highp float;
uniform vec4 u_color;
out vec4 o_color;

void main()
{
    o_color = u_color;
}
