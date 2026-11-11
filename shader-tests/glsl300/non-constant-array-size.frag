#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.1.9 - array sizes must be constant integral expressions
// @feature: GLSL300.TYPES.ARRAY
// @feature: GLSL300.QUAL.CONST
precision mediump float;
uniform int u_n;
out vec4 o;
void main() { float a[u_n]; o = vec4(1.0); }
