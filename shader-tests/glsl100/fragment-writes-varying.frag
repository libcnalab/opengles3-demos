#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §4.3.5 "A fragment shader can not write to a varying variable"
// @feature: GLSL100.ATTRIBUTE_VARYING
precision mediump float;
varying vec4 v_color;
void main() { v_color = vec4(1.0); gl_FragColor = v_color; }
