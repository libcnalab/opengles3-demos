#version 300 es
// The test chart that every read and copy in this demo starts from. Pixel (x, y) of the 64x64
// framebuffer gets the value (4x, 4y, checker, 128 + x + y) in 8-bit steps, so that every pixel has
// a distinct, position-dependent RGBA value (red grows along x, green along y, alpha along both,
// blue is an 8x8 checkerboard). Reading or copying the wrong place, flipping rows, or swapping
// channels all change the values. gl_FragCoord is in window coordinates: the pixel (x, y) has its
// center at (x + 0.5, y + 0.5), so ivec2(gl_FragCoord.xy) is the pixel index.
precision highp float;
precision highp int;
out vec4 o_color;
void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    float checker = (((p.x >> 3) ^ (p.y >> 3)) & 1) == 1 ? 204.0 : 51.0;
    o_color = vec4(float(4 * p.x), float(4 * p.y), checker, float(128 + p.x + p.y)) / 255.0;
}
