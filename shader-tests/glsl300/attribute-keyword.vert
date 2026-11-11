#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.8 Keywords - attribute and varying are reserved; using them is an error
// @feature: GLSL300.VERSION
attribute vec4 a_position;
void main() { gl_Position = a_position; }
