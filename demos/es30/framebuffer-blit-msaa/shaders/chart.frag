#version 300 es
// The blit source chart: three RGBA8 attachments of the same 64 x 64 target, each a different
// permutation of one pattern, so that glReadBuffer can be told apart by looking at the copy.
//   red   grows by 4 per pixel along x, green by 4 per pixel along y (every pixel is unique),
//   blue  is an 8 x 8 checkerboard (sharp edges: they make LINEAR and NEAREST distinguishable).
precision highp float;
layout(location = 0) out vec4 o_chart0;
layout(location = 1) out vec4 o_chart1;
layout(location = 2) out vec4 o_chart2;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);  // window coordinates of the pixel, 0..63
    float r = float(4 * p.x) / 255.0;
    float g = float(4 * p.y) / 255.0;
    float b = (((p.x >> 3) + (p.y >> 3)) & 1) == 1 ? 200.0 / 255.0 : 60.0 / 255.0;
    o_chart0 = vec4(r, g, b, 1.0);
    o_chart1 = vec4(b, r, g, 1.0);
    o_chart2 = vec4(g, b, r, 1.0);
}
