#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 ternary: "This operator evaluates the first expression, which must result in a scalar Boolean"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { int i = 1; float f = i ? 1.0 : 0.0; o = vec4(f); }
