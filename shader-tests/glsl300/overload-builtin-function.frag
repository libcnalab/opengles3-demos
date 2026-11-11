#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.3 "overloading or redefining built-in functions is an error"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
float sin(int x) { return float(x); }
void main() { o = vec4(sin(1)); }
