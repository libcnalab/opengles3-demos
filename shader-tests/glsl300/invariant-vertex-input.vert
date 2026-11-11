#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "Only variables output from a shader can be candidates for invariance"; §4.8 "invariant in float x; // Error. An input cannot be invariant."
// @feature: GLSL300.INVARIANT
invariant in vec4 a_position;
void main() { gl_Position = a_position; }
