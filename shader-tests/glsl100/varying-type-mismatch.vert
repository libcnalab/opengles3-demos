#version 100
// @expect: link-fail
// @spec: GLSL ES 1.00 §4.3.5 "The type of varying variables with the same name declared in both the vertex and fragment shaders must match, otherwise the link command will fail"
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute vec4 a_position;
varying vec3 v_color;
void main() { v_color = a_position.xyz; gl_Position = a_position; }
