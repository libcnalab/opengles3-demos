#version 300 es
// highp keeps the k / 255 face colors exact on their way to the 8-bit color buffer.
precision highp float;

flat in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = v_color;
}
