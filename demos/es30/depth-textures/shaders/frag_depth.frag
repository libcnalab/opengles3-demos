#version 300 es
// Writes a chosen value per column into a 16x1 depth texture through gl_FragDepth.
// ES 3.0 §3.9.2.3: for fixed-point depth buffers the value is clamped to [0, 1] and converted to
// fixed point like a window z (§2.13.1, §2.1.6.2); for floating-point buffers it is only clamped.
precision highp float;

uniform float u_depth[16];

void main()
{
    int i = clamp(int(gl_FragCoord.x), 0, 15);
    gl_FragDepth = u_depth[i];
}
