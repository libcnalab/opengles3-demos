#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §8.7 "Functions containing the bias parameter are available only in the fragment shader"
// @feature: GLSL100.TEXTURE2D
attribute vec4 a_position;
uniform sampler2D u_tex;
varying vec4 v_color;
void main() { v_color = texture2D(u_tex, a_position.xy, 1.0); gl_Position = a_position; }
