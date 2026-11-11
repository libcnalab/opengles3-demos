#version 300 es
// Shows one 4x4 result block of the RGBA32UI probe target, magnified to fill a grid cell. The block is read with
// texelFetch from a usampler2D; u_kind says how the 32-bit words are interpreted.
precision highp float;
precision highp int;

uniform highp usampler2D u_results;
uniform ivec2 u_block;  // lower-left texel of the block in the probe target
uniform int u_kind;     // 0 RGBA8 lookup (float bits), 1 RGBA32UI/RGBA8UI lookup, 2 RGBA32I lookup, 3 shadow pair,
                        // 4 textureSize, 5 derivatives
uniform vec4 u_rect;    // the cell in window coordinates: x, y, width, height
out vec4 o_color;

const vec3 kPalette[8] = vec3[8](vec3(0.902, 0.235, 0.204), vec3(0.957, 0.612, 0.180), vec3(0.941, 0.886, 0.290),
                                 vec3(0.384, 0.769, 0.329), vec3(0.231, 0.702, 0.769), vec3(0.290, 0.416, 0.875),
                                 vec3(0.588, 0.341, 0.839), vec3(0.878, 0.878, 0.878));

vec3 texel_colour(int level, int i, int j)
{
    return kPalette[clamp(level, 0, 7)] * (((i + j) & 1) == 0 ? 1.0 : 0.75);
}

void main()
{
    vec2 local = (gl_FragCoord.xy - u_rect.xy) / u_rect.zw;
    ivec2 t = clamp(ivec2(floor(local * 4.0)), ivec2(0), ivec2(3));
    uvec4 r = texelFetch(u_results, u_block + t, 0);
    vec4 f = uintBitsToFloat(r);
    vec3 c;
    if (u_kind == 0)
        c = texel_colour(int(round((f.r * 255.0 - 16.0) / 32.0)), int(round(f.g * 255.0 / 8.0)), int(round(f.b * 255.0 / 8.0)));
    else if (u_kind == 1)
        c = texel_colour(int(r.x), int(r.y), int(r.z));
    else if (u_kind == 2)
        c = texel_colour(-1 - int(r.x), -1 - int(r.y), -1 - int(r.z));
    else if (u_kind == 3)
        c = vec3(0.15 + 0.7 * f.y, 0.15 + 0.7 * f.x, 0.2);  // (1, 0) = the expected texel passed one test and failed the other
    else if (u_kind == 4)
        c = vec3(float(r.x) / 32.0, float(r.y) / 16.0, float(r.z) / 8.0);
    else
        c = clamp(abs(f.xyz) / 8.0, 0.0, 1.0);

    // A thin dark frame separates the cells.
    vec2 edge = min(local, 1.0 - local) * u_rect.zw;
    if (min(edge.x, edge.y) < 1.0) c *= 0.35;
    o_color = vec4(c, 1.0);
}
