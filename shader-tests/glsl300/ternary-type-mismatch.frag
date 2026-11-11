#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 ternary: "The second and third expressions can be any type ... as long their types match"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { bool c = true; float f = c ? 1 : 1.0; o = vec4(f); }
