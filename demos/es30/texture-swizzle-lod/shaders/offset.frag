#version 300 es
// Texel offsets (GLSL ES 3.00 8.8, ES 3.0 3.8.10.1 equation 3.19): the offset must be a CONSTANT
// EXPRESSION, so the limits MIN_PROGRAM_TEXEL_OFFSET / MAX_PROGRAM_TEXEL_OFFSET queried by the
// application are pasted in as OFF_MIN and OFF_MAX (two #define lines inserted after #version) before
// the shader is compiled. Five different constant offsets are compiled in; a uniform picks one.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform int u_func;        // 0 texelFetchOffset, 1 textureOffset, 2 textureProjOffset, 3 textureLodOffset, 4 textureGradOffset
uniform int u_off;         // which of the five constant offsets below
uniform float u_lod;       // level for textureLodOffset / texelFetchOffset
uniform vec2 u_dx;         // textureGradOffset gradients
uniform vec2 u_dy;
uniform float u_q;         // projective coordinate for textureProjOffset
uniform ivec2 u_origin;    // lower left window pixel of the viewport: pixel (x, y) of the cell = gl_FragCoord - u_origin
uniform int u_size;        // edge of the (square) level that is being sampled

in vec2 v_corner;
out vec4 o_color;

// One lookup function per constant offset. Pixel (x, y) of the cell is the centre of texel (x, y) of the
// sampled level, so every plain lookup returns the texel at (x, y) + offset (after the wrap mode). The
// texelFetchOffset variant instead fetches the MIRRORED texel (size - 1 - x, y): its coordinate is chosen as
// P = mirrored - offset, so that P + offset is always inside the level (anything else is undefined, 2.12.9.2),
// while P itself runs far outside it for large offsets.
#define DEFINE_LOOKUP(NAME, OFFSET)                                                                    \
    vec4 NAME(ivec2 q, vec2 uv)                                                                        \
    {                                                                                                  \
        if (u_func == 0)                                                                               \
            return texelFetchOffset(u_tex, ivec2(u_size - 1 - q.x, q.y) - OFFSET, int(u_lod), OFFSET); \
        if (u_func == 1)                                                                               \
            return textureOffset(u_tex, uv, OFFSET);                                                   \
        if (u_func == 2)                                                                               \
            return textureProjOffset(u_tex, vec3(uv * u_q, u_q), OFFSET);                              \
        if (u_func == 3)                                                                               \
            return textureLodOffset(u_tex, uv, u_lod, OFFSET);                                         \
        return textureGradOffset(u_tex, uv, u_dx, u_dy, OFFSET);                                       \
    }

DEFINE_LOOKUP(lookup_small, ivec2(1, 2))
DEFINE_LOOKUP(lookup_min_min, ivec2(OFF_MIN, OFF_MIN))
DEFINE_LOOKUP(lookup_max_max, ivec2(OFF_MAX, OFF_MAX))
DEFINE_LOOKUP(lookup_min_max, ivec2(OFF_MIN, OFF_MAX))
DEFINE_LOOKUP(lookup_max_min, ivec2(OFF_MAX, OFF_MIN))

void main()
{
    ivec2 q = ivec2(gl_FragCoord.xy) - u_origin;
    switch (u_off)
    {
    case 0: o_color = lookup_small(q, v_corner); break;
    case 1: o_color = lookup_min_min(q, v_corner); break;
    case 2: o_color = lookup_max_max(q, v_corner); break;
    case 3: o_color = lookup_min_max(q, v_corner); break;
    default: o_color = lookup_max_min(q, v_corner); break;
    }
}
