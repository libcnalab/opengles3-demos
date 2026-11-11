#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "The invariant qualifier must appear before any interpolation qualifiers or storage qualifiers"; §4.7 Order of Qualification
// @feature: GLSL300.INVARIANT
in vec4 a_position;
centroid invariant out vec4 v_a;
void main() { v_a = a_position; gl_Position = a_position; }
