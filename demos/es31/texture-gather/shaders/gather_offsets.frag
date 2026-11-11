#version 320 es
// textureGatherOffsets exists only in GLSL ES 3.20 (section 8.9.4 there; the 3.10 specification has textureGather and
// textureGatherOffset only). The demo compiles this shader only on a context that reports ES 3.2.
//
// The four texels of the result are no longer the 2 x 2 footprint: texel k is the T_i0_j0 texel of the footprint that
// the offset kOffsets[k] selects. The offsets must be constant integral expressions.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform float u_size;

out vec4 o_value;

const ivec2 kOffsets[4] = ivec2[4](ivec2(-1, 0), ivec2(3, 1), ivec2(0, -2), ivec2(2, 2));

void main()
{
    vec2 st = vec2(ivec2(gl_FragCoord.xy)) / u_size;
    o_value = textureGatherOffsets(u_tex, st, kOffsets, 2);
}
