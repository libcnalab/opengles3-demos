#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §4.3.3 "It is an error to declare an attribute variable in any type of shader other than a vertex shader"
// @feature: GLSL100.ATTRIBUTE_VARYING
precision mediump float;
attribute vec4 a_color;
void main() { gl_FragColor = a_color; }
