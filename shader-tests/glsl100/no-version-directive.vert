// No #version line at all: the shader must be compiled as GLSL ES 1.00.
// @expect: link-pass
// @spec: GLSL ES 1.00 §3.4 "compilation units that do not include a #version directive will be treated as targeting version 100"; GLSL ES 3.00 §3.4 (same rule for ES 3.0 compilers)
// @feature: GLSL100.VERSION
#if __VERSION__ != 100
#error "a shader without #version must see __VERSION__ == 100"
#endif
attribute vec4 a_position;
varying vec4 v_color;
void main() { v_color = a_position; gl_Position = a_position; }
