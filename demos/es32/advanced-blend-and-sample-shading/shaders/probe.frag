#version 320 es
// Writes one sample-related quantity per pass into an RGBA8 multisample texture. main.cpp inserts "#define MODE n"
// after the #version line. Positions are stored with 16 bits of precision: t in [0, 1] becomes two 8-bit channels
// (floor(255 t) / 255, fract(255 t)), which the CPU decodes as (hi + lo / 255) / 255.
//
//   MODE 0  gl_SampleID, gl_NumSamples, gl_SampleMaskIn[0]        (GLSL ES 3.20 section 7.1.5)
//   MODE 1  gl_SamplePosition
//   MODE 2  a "sample in" input                                    (GLSL section 4.5, ES 3.2 section 14.1)
//   MODE 3  an ordinary input (per sample only through MinSampleShading)
//   MODE 4  interpolateAtSample(v_pos, u_index)                    (GLSL section 8.14.2)
//   MODE 5  interpolateAtOffset(v_pos, u_offset)
//   MODE 6  interpolateAtCentroid(v_pos)
//   MODE 7  a "centroid in" input
//   MODE 8  gl_SampleMaskIn[0] of a pixel-rate shader (alpha 1 marks a written sample)
precision highp float;
precision highp int;

in highp vec2 v_pos;
#if MODE == 2
sample in highp vec2 v_spos;
#elif MODE == 7
centroid in highp vec2 v_cpos;
#endif

uniform float u_scale;    // positions are divided by this before they are encoded
uniform int u_index;
uniform vec2 u_offset;

out vec4 o_value;

vec2 enc(float t)
{
    float s = clamp(t, 0.0, 1.0) * 255.0;
    float hi = floor(s);
    return vec2(hi / 255.0, s - hi);
}

vec4 enc2(vec2 v)
{
    vec2 t = v / u_scale;
    return vec4(enc(t.x), enc(t.y));
}

void main()
{
#if MODE == 0
    o_value = vec4(float(gl_SampleID), float(gl_NumSamples), float(gl_SampleMaskIn[0] & 255), 255.0) / 255.0;
#elif MODE == 1
    o_value = vec4(enc(gl_SamplePosition.x), enc(gl_SamplePosition.y));
#elif MODE == 2
    o_value = enc2(v_spos);
#elif MODE == 3
    o_value = enc2(v_pos);
#elif MODE == 4
    o_value = enc2(interpolateAtSample(v_pos, u_index));
#elif MODE == 5
    o_value = enc2(interpolateAtOffset(v_pos, u_offset));
#elif MODE == 6
    o_value = enc2(interpolateAtCentroid(v_pos));
#elif MODE == 7
    o_value = enc2(v_cpos);
#else
    o_value = vec4(float(gl_SampleMaskIn[0] & 255) / 255.0, 0.0, 0.0, 1.0);
#endif
}
