#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.8 "The ternary operator (?:) is also not allowed as an l-value"
// @xfail: Mesa | Mesa 25.0 accepts a parenthesised ?: expression as the left side of an assignment
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { float a = 0.0; float b = 0.0; bool c = true; (c ? a : b) = 1.0; o = vec4(a, b, 0.0, 1.0); }
