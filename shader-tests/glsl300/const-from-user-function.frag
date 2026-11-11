#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.3 "Function calls to user-defined functions (non-built-in functions) cannot be used to form constant expressions"
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
float one() { return 1.0; }
void main() { const float c = one(); o = vec4(c); }
