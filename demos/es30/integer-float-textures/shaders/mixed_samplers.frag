#version 300 es
// Negative case, never drawn successfully: two ACTIVE samplers of different types (sampler2D and
// usampler2D). The program links. If both uniforms keep the value 0 (the same texture image unit),
// the first draw call fails with INVALID_OPERATION (ES 3.0 2.12.9.7 Validation: "any two active
// samplers in the current program object are of different types, but refer to the same texture
// image unit"). With two different units the same program draws normally.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

uniform sampler2D u_float_tex;
uniform usampler2D u_uint_tex;

layout(location = 0) out uvec4 o_value;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    o_value = floatBitsToUint(texelFetch(u_float_tex, p, 0)) + texelFetch(u_uint_tex, p, 0);
}
