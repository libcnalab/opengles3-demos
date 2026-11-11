#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1.1 "expressions that do not evaluate to l-values cannot be passed to parameters declared as out or inout"
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
void f(out float x) { x = 1.0; }
void main() { float a = 0.0; f(a + 1.0); o = vec4(a); }
