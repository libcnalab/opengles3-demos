#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1 "vec4 f(in vec4 x, in ivec4 y); // error, only qualifier differs"
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
void f(in vec4 x, out vec4 y) { y = x; }
void f(in vec4 x, in vec4 y) { }
void main() { vec4 a = vec4(1.0); vec4 b; f(a, b); o = b; }
