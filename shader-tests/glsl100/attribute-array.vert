#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §4.3.3 "Attribute variables cannot be declared as arrays or structures"
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute vec4 a_position[2];
void main() { gl_Position = a_position[0] + a_position[1]; }
