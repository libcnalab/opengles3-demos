#version 300 es
// Diagonal stripes in window coordinates: the stripe index is floor((x + y) / period), and
// even/odd stripes alternate between two colors. The period is a power of two so the division is
// exact, which lets the CPU reference predict every pixel.
precision highp float;
uniform vec4 u_color_a;
uniform vec4 u_color_b;
uniform float u_period;
layout(location = 0) out vec4 o_color;

void main()
{
    float band = floor((gl_FragCoord.x + gl_FragCoord.y) / u_period);
    o_color = (mod(band, 2.0) < 0.5) ? u_color_a : u_color_b;
}
