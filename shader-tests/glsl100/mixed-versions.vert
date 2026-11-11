#version 100
// @expect: link-fail
// @spec: GLSL ES 3.00 §3.4 "Shaders declaring version 3.00 ... cannot be linked with shaders declaring version 1.00"
// @feature: GLSL100.MIX_VERSIONS
attribute vec4 a_position;
void main() { gl_Position = a_position; }
