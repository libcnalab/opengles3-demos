#version 99
// @expect: compile-fail
// @spec: GLSL ES 1.00 §3.4 "Any number less than 100 will cause an error to be generated"
// @feature: GLSL100.VERSION
attribute vec4 a_position;
void main() { gl_Position = a_position; }
