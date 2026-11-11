#version 300 es
// Displayed colour of a low-poly facet: palette entry times shade, both taken from the provoking vertex.
precision highp float;

flat in uint v_palette;
flat in float v_shade;

uniform vec3 u_palette[8];

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(u_palette[int(v_palette)] * v_shade, 1.0);
}
