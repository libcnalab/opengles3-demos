#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.1.10 Implicit Conversions - there are no implicit conversions
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
void main() { float f = 1; o = vec4(f); }
