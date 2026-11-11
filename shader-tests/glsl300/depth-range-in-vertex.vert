#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §7.4 uniform gl_DepthRangeParameters gl_DepthRange { highp float near, far, diff } is available to the vertex language
// @feature: GLSL300.BUILTIN_UNIFORMS
in vec4 a_position;
out highp vec3 v_range;
void main() { v_range = vec3(gl_DepthRange.near, gl_DepthRange.far, gl_DepthRange.diff); gl_Position = a_position; }
