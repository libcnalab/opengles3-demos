#version 320 es
// @expect: compile-pass
// @spec: GLSL ES 3.20 §7.1.3 gl_SampleID/gl_SamplePosition/gl_SampleMaskIn/gl_SampleMask, §8.13 interpolateAt*
// @feature: GLSL320.SAMPLE_VARS
// @feature: GLSL320.FN.INTERPOLATE_AT
precision mediump float;
sample in vec2 v_uv;
out vec4 o;
void main()
{
    vec2 c = interpolateAtCentroid(v_uv) + interpolateAtOffset(v_uv, vec2(0.25)) + interpolateAtSample(v_uv, 0);
    gl_SampleMask[0] = gl_SampleMaskIn[0];
    o = vec4(c + gl_SamplePosition, float(gl_SampleID), 1.0);
}
