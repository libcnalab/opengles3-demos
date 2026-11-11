#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.3 "Names of built-in functions cannot be redeclared as functions"; §6.1 "A shader cannot redefine or overload built-in functions"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
float sin(float x) { return x; }
void main() { o = vec4(sin(1.0)); }
