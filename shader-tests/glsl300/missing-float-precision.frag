#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.5.4 - the fragment language has no default precision for float
// @feature: GLSL300.PRECISION.DEFAULT
out vec4 o;
void main() { o = vec4(1.0); }
