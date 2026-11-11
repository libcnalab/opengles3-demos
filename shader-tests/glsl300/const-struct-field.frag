#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.2 "Structure fields may not be qualified with const"
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
struct S { const float x; };
void main() { o = vec4(1.0); }
