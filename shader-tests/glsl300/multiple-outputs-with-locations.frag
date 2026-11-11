#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.3.8.2 Output Layout Qualifiers
// @feature: GLSL300.FRAG.MRT
// @feature: GLSL300.LAYOUT.LOCATION
precision mediump float;
layout(location = 0) out vec4 o_a;
layout(location = 1) out vec4 o_b;
void main() { o_a = vec4(1.0); o_b = vec4(0.0); }
