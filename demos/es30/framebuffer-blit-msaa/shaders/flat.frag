#version 300 es
// One uniform color. Used for the depth/stencil scenes, their probe passes and the triangle:
// those only care where a fragment is written, not what it looks like.
precision highp float;
uniform vec4 u_color;
out vec4 o_color;

void main()
{
    o_color = u_color;
}
