#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4.6.1 "The invariant keyword can be followed by a comma separated list of previously declared identifiers"; §9 grammar: single_declaration "INVARIANT IDENTIFIER" continued by init_declarator_list "COMMA IDENTIFIER"
// @feature: GLSL300.INVARIANT
// @xfail: ANGLE | ANGLE 2.1.17841 reports "',' : syntax error" for "invariant v_a, v_b;" (each name in its own invariant statement is accepted)
in vec4 a_position;
out vec4 v_a;
out vec4 v_b;
invariant v_a, v_b;
void main() { v_a = a_position; v_b = a_position * 2.0; gl_Position = a_position; }
