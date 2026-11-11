#version 300 es
// Three RGBA8 attachments written at once (the framebuffer whose attachments are invalidated):
//   red   grows by 4 per pixel along x, green by 4 per pixel along y (every pixel is unique),
//   blue  is an 8 x 8 checkerboard; the three outputs are permutations of the same three channels.
precision highp float;
layout(location = 0) out vec4 o_color0;
layout(location = 1) out vec4 o_color1;
layout(location = 2) out vec4 o_color2;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);  // 0..63
    float r = float(4 * p.x) / 255.0;
    float g = float(4 * p.y) / 255.0;
    float b = (((p.x >> 3) + (p.y >> 3)) & 1) == 1 ? 200.0 / 255.0 : 60.0 / 255.0;
    o_color0 = vec4(r, g, b, 1.0);
    o_color1 = vec4(b, r, g, 1.0);
    o_color2 = vec4(g, b, r, 1.0);
}
