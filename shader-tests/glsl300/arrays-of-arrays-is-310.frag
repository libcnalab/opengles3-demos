#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.1.9 Arrays - only one-dimensional arrays (arrays of arrays: GLSL ES 3.10)
// @feature: GLSL300.TYPES.ARRAY
precision mediump float;
out vec4 o;
void main() { float a[2][2]; a[0][0] = 1.0; o = vec4(a[0][0]); }
