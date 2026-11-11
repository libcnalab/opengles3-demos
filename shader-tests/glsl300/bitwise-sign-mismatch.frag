#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 bitwise &, ^, |: "The fundamental types of the operands (signed or unsigned) must match"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { uint u = 6u & 3; o = vec4(float(u)); }
