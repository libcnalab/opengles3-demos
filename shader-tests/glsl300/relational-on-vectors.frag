#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 "The relational operators ... operate only on scalar integer and scalar floating-point expressions"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { bool b = vec2(1.0) < vec2(2.0); o = vec4(float(b)); }
