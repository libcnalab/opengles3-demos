#version 300 es
// highp is available in every ES 3.0 fragment shader and keeps the interpolated constant colors
// exact, so the CPU comparison needs no allowance for 10-bit mediump arithmetic.
precision highp float;

in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = clamp(v_color, 0.0, 1.0);  // values outside [0, 1] saturate, as in the framebuffer
}
