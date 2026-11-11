#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.1.2 "Expressions used for conditional jumps (if, for, ?:, while, do-while) must evaluate to the type bool"
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
void main() { int i = 1; if (i) { o = vec4(1.0); } else { o = vec4(0.0); } }
