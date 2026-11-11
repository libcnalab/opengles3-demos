#version 300 es
// @expect: build-fail
// @spec: ES 3.0 §3.9.2.3 "conflicting binding assignments" and GLSL ES 3.00 §4.3.8.2 - two outputs on one location is an error (Mesa rejects it at link time)
// @feature: GLSL300.FRAG.MRT
precision mediump float;
layout(location = 1) out vec4 o_a;
layout(location = 1) out vec4 o_b;
void main() { o_a = vec4(1.0); o_b = vec4(0.0); }
