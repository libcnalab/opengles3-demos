#version 300 es
// One constant color (u_color), used for the blended overlay and for the default-framebuffer patch.
precision highp float;

uniform vec4 u_color;
out vec4 o_color;

void main()
{
    o_color = u_color;
}
