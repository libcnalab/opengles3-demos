#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.3 "the sequence operator ( , ) and the assignment operators ( =, +=, ...) are not included in the operators that can create a constant expression"
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
void main() { const int c = (1, 2); o = vec4(float(c)); }
