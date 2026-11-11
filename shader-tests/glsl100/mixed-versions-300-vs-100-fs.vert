#version 300 es
// @expect: link-fail
// @spec: GLSL ES 3.00 §3.4 "Shaders declaring version 3.00 of the shading language cannot be linked with shaders declaring version 1.00"; §1.5
// @feature: GLSL100.MIX_VERSIONS
in vec4 a_position;
void main() { gl_Position = a_position; }
