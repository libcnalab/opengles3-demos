#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.3 "void f(int); struct f {int x;}; // Error: type 'f' conflicts with function 'f'"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
void f(int);
struct f { int x; };
void main() { o = vec4(1.0); }
