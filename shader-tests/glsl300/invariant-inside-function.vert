#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "All uses of invariant must be at the global scope"
// @feature: GLSL300.INVARIANT
in vec4 a_position;
out vec4 v_a;
void main() { invariant v_a; v_a = a_position; gl_Position = a_position; }
