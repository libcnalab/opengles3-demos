#version 300 es
// @expect: link-pass
// @spec: GLSL ES 3.00 §4.6.1 "an invariant output from one shader stage will still match an input of a subsequent stage without the input being declared as invariant"
// @feature: GLSL300.INVARIANT
in vec4 a_position;
invariant out vec4 v_color;
void main() { v_color = a_position; gl_Position = a_position; }
