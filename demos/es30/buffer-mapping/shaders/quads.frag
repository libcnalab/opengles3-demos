#version 300 es
// highp keeps the interpolation of equal 8-bit colors exact enough that the 8-bit framebuffer
// value equals the byte that was written into the buffer.
precision highp float;

in vec4 v_color;
out vec4 o_color;

void main()
{
    o_color = v_color;
}
