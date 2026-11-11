#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §4.3.5 "The varying qualifier can be used only with the data types float, vec2, vec3, vec4, mat2, mat3, and mat4, or arrays of these"
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute vec4 a_position;
varying int v_index;
void main() { v_index = 1; gl_Position = a_position; }
