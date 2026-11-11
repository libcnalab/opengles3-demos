#version 300 es
// @expect: link-fail
// @spec: GLSL ES 3.00 §4.3.10 - a fragment input that is statically used must have a matching vertex output
// @feature: GLSL300.INTERFACE_MATCHING
void main() { gl_Position = vec4(0.0); }
