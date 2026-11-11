#version 100
// @expect: link-pass
// @spec: GLSL ES 1.00 §7.2 "Fragment shaders output values ... using the built-in variables gl_FragColor and gl_FragData"; GLSL ES 3.00 §1.5 (only gl_FragData[0] is usable on ES 3.0)
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute vec4 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = a_position; }
