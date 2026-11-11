#version 100
// @expect: link-pass
// @spec: GLSL ES 1.00 §4.3.5 "The precision of varying variables does not need to match"
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute vec4 a_position;
varying highp vec4 v_color;
void main() { v_color = a_position; gl_Position = a_position; }
