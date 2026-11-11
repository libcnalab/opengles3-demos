#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1 "int f(in vec4 x, out ivec4 y); // error, only return type differs"
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
float f(vec4 x) { return x.x; }
int f(vec4 x) { return 1; }
void main() { o = vec4(f(vec4(1.0))); }
