#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.2 "A function's parameter declarations and body together form a single scope" - int f(int k) { int k = k + 3; } is a redeclaration error
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
int f(int k) { int k = k + 3; return k; }
void main() { o = vec4(float(f(1))); }
