#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1.1 "the const qualifier cannot be used with out or inout"
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
void f(const out float x) { }
void main() { float a = 0.0; f(a); o = vec4(a); }
