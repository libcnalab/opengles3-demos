#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.8 "The lvalue-expression and rvalue-expression must have the same type. Any type-conversions must be specified explicitly via constructors"
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
void main() { int i = 1u; o = vec4(float(i)); }
