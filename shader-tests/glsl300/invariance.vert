#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.6 Variance and the Invariant Qualifier
// @feature: GLSL300.INVARIANT
#pragma STDGL invariant(all)
invariant gl_Position;
in vec4 a_position;
void main() { gl_Position = a_position; }
