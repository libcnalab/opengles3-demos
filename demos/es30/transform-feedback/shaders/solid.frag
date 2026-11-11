#version 300 es
precision highp float;
// Used by every capture pass. The capture passes run with RASTERIZER_DISCARD enabled, so this shader must
// never produce a pixel; the color is a loud red so that a leak would be visible and compared.
uniform vec4 u_color;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_color;
}
