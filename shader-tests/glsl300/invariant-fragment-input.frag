#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "Only variables output from a shader can be candidates for invariance"; §4.8 "invariant in float x; // Error. An input cannot be invariant."
// @feature: GLSL300.INVARIANT
precision mediump float;
invariant in vec4 v_color;
out vec4 o;
void main() { o = v_color; }
