#version 310 es
// Five fragment-shader passes over the same r32ui image, one thread of work per pixel, each pass depending only on the
// value that the previous pass left at ITS OWN pixel (gl_FragCoord). That is the dependency pattern for which
// glMemoryBarrierByRegion(SHADER_IMAGE_ACCESS_BARRIER_BIT) is sufficient (ES 3.2 7.11.2: the region may be as small as
// one framebuffer pixel). Pass 0 initializes, passes 1..3 read-modify-write, pass 4 turns the value into a color.
precision highp float;
precision highp int;
layout(r32ui, binding = 0) uniform highp uimage2D u_img;  // read-write: allowed for r32ui
uniform int u_pass;
layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    uint x = uint(p.x);
    uint y = uint(p.y);
    if (u_pass == 0)
    {
        imageStore(u_img, p, uvec4(x * 3u + y * 5u + 1u));
        o_color = vec4(0.0);
        return;
    }
    uint v = imageLoad(u_img, p).x;
    if (u_pass < 4)
    {
        uint k = uint(u_pass);
        v += k * 16u + ((x >> 3u) & 1u) * k * 64u;
        imageStore(u_img, p, uvec4(v));
        o_color = vec4(0.0);
        return;
    }
    o_color = vec4(float(v % 1024u) / 1023.0, float((v / 4u) % 256u) / 255.0, float((v >> 3u) & 1u) * 0.8 + 0.1, 1.0);
}
