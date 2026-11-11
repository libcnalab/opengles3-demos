#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.3 "int x; int x; // Error: repeated variable definition"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
float x;
float x;
void main() { o = vec4(x); }
