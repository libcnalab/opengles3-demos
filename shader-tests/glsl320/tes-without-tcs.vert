#version 320 es
// @expect: link-fail
// @spec: ES 3.2 §7.3 - a non-separable program with a tessellation evaluation shader but no control shader fails to link
// @feature: GLSL320.STAGE_COMBINATIONS
in vec4 a_position;
void main() { gl_Position = a_position; }
