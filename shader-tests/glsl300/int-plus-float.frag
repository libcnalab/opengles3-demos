#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4 (chapter introduction) "There are no implicit conversions between types"; §5.9 arithmetic operands must have matching types
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
void main() { float f = 1.0 + 1; o = vec4(f); }
