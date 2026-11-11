#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.5.4 - the vertex language predeclares precision highp float/int
// @feature: GLSL300.PRECISION.DEFAULT
// @feature: GLSL300.PRECISION.QUALIFIERS
in vec4 a_position;
out lowp vec4 v_color;
void main() { mediump float m = 0.5; v_color = vec4(m); gl_Position = a_position; }
