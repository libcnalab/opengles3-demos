#version 310 es
// @expect: link-fail
// @spec: GLSL ES 3.10 §3.3 - shaders declaring version 3.10 cannot be linked with shaders declaring a previous version
// @feature: GLSL310.VERSION
in vec4 a_position;
void main() { gl_Position = a_position; }
