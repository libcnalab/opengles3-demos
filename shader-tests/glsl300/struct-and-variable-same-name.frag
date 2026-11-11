#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.3 "struct f {int x;}; int f; // Error: conflicts with the type 'f'"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
struct f { int x; };
int f;
void main() { o = vec4(1.0); }
