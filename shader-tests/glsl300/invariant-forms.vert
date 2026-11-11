#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.6.1 invariant on a previously declared output ("invariant Color;"), on gl_Position, and as part of a declaration ("invariant centroid out vec3 Color;"); the comma-separated list is the separate case invariant-comma-list
// @feature: GLSL300.INVARIANT
in vec4 a_position;
out vec4 v_a;
out vec4 v_b;
invariant v_a;
invariant v_b;
invariant centroid out vec4 v_c;
invariant gl_Position;
void main() { v_a = a_position; v_b = a_position * 2.0; v_c = a_position.wzyx; gl_Position = a_position; }
