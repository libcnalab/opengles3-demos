#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.6.1 "invariant Color_2; // error: Color_2 has not been declared"
// @feature: GLSL300.INVARIANT
in vec4 a_position;
invariant v_missing;
void main() { gl_Position = a_position; }
