#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.2 "For both for and while loops, the sub-statement itself does not introduce a new scope for variable names"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
void main() { float s = 0.0; for (int i = 0; i < 2; i++) { int i = 5; s += float(i); } o = vec4(s); }
