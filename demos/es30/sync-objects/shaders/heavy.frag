#version 300 es
precision highp float;
// Keeps the GPU busy for a while so that fences are not signaled the instant they are created. The result of
// the loop is multiplied by u_zero, which is 0.0 at run time, so the output is exactly u_color.
uniform int u_iterations;
uniform float u_zero;
uniform vec4 u_color;
layout(location = 0) out vec4 o_color;

void main()
{
    float acc = 0.0;
    for (int i = 0; i < u_iterations; ++i)
        acc += sin(float(i) + gl_FragCoord.x * 0.015625);
    o_color = u_color + vec4(acc * u_zero);
}
