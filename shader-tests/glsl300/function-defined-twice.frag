#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.3 "void f(int) {...} void f(int) {...} // Error: repeated definition"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
float f(int a) { return 1.0; }
float f(int a) { return 2.0; }
void main() { o = vec4(f(1)); }
