#version 300 es
// @expect: build-fail
// @spec: GLSL ES 3.00 §4.6.1 "All uses of invariant must be at the global scope, and before any use of the variables being declared as invariant"; §1.3: a "must" violation has to be reported, but "All lexical, grammatical and semantic errors must have been detected following a call to glLinkProgram" - compile or link time
// @feature: GLSL300.INVARIANT
// @xfail: ANGLE | ANGLE 2.1.17841 compiles and links "invariant v_a;" placed after a function that already wrote v_a
in vec4 a_position;
out vec4 v_a;
void set_a() { v_a = a_position; }
invariant v_a;
void main() { set_a(); gl_Position = a_position; }
