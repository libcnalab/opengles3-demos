#version 300 es
// @expect: build-fail
// @spec: GLSL ES 3.00 §4.3.8.2 - with more than one output, every output must have a location
// @feature: GLSL300.FRAG.MRT
// @feature: GLSL300.LAYOUT.LOCATION
precision mediump float;
out vec4 o_a;
out vec4 o_b;
void main() { o_a = vec4(1.0); o_b = vec4(0.0); }
