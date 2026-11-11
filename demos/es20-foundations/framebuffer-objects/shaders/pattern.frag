#version 300 es
// First pass of render-to-texture: a procedural image whose value at every pixel is known exactly
// on the CPU. Red ramps with x, green ramps with y (both 0..1 across the render target), blue is an
// 8x8-pixel checkerboard. gl_FragCoord is in window coordinates, so pixel (i, j) has the center
// (i + 0.5, j + 0.5) and the integer pixel index is its floor.
precision highp float;
precision highp int;
uniform float u_size;  // edge length of the (square) render target in pixels
out vec4 o_color;
void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    float checker = (((p.x >> 3) ^ (p.y >> 3)) & 1) == 1 ? 0.8 : 0.2;
    o_color = vec4(vec2(p) / (u_size - 1.0), checker, 1.0);
}
