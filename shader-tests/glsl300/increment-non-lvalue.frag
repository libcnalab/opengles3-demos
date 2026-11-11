#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 "For post- and pre-increment and decrement, the expression must be one that could be assigned to (an l-value)"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { int a = 1; int b = 2; int c = (a + b)++; o = vec4(float(c)); }
