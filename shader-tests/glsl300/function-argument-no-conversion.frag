#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1 "When function calls are resolved, an exact type match for all the arguments is required"
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
float twice(float x) { return 2.0 * x; }
void main() { o = vec4(twice(2)); }
