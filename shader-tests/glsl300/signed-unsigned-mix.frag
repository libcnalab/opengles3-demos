#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 "If the operands are integral types, they must both be signed or both be unsigned"
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
void main() { uint u = 1u + 1; o = vec4(float(u)); }
