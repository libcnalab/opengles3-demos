#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "All uses of invariant must be at the global scope, and before any use of the variables being declared as invariant"
// @feature: GLSL300.INVARIANT
in vec4 a_position;
out vec4 v_a;
void set_a() { v_a = a_position; }
invariant v_a;
void main() { set_a(); gl_Position = a_position; }
