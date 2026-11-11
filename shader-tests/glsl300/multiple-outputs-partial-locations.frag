#version 300 es
// @expect: compile-fail
// @spec: ES 3.0 §3.9.2.3 and GLSL ES 3.00 §4.3.8.2 - with more than one output, every output needs a location
// @xfail: Mesa | Mesa 25.0 accepts one located and one unlocated fragment output
// @feature: GLSL300.FRAG.MRT
precision mediump float;
layout(location = 0) out vec4 o_a;
out vec4 o_b;
void main() { o_a = vec4(1.0); o_b = vec4(0.0); }
