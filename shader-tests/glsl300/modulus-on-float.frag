#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 "The operator modulus (%) is not defined for any other data types (non-integral types)"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { float f = 5.5 % 2.0; o = vec4(f); }
