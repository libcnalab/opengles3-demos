#version 320 es
// @expect: link-pass
// @spec: GLSL ES 3.20 §4.4.1.1 (TES layout), §4.4.2.1 (TCS vertices), §7.1.5-7.1.6 tessellation built-ins
// @feature: GLSL320.TESS.CONTROL
// @feature: GLSL320.TESS.EVAL
in vec4 a_position;
void main() { gl_Position = a_position; }
