#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.2 "It is an error to write to a const variable outside of its declaration"
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
void main() { const float c = 1.0; c = 2.0; o = vec4(c); }
