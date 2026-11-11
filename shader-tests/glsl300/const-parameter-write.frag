#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1.1 "Only the function's copy is modified. This can be prevented by declaring a parameter with the const qualifier"; §4.3 const parameter "is read-only"
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
float f(const float x) { x = 2.0; return x; }
void main() { o = vec4(f(1.0)); }
