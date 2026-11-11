#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.8 - layout(binding) does not exist before GLSL ES 3.10
// @feature: GLSL300.VERSION
precision mediump float;
layout(binding = 0) uniform sampler2D u_tex;
out vec4 o;
void main() { o = texture(u_tex, vec2(0.5)); }
