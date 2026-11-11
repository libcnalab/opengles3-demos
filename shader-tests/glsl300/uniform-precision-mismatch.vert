#version 300 es
// @expect: link-fail
// @spec: GLSL ES 3.00 §4.5.3 - the same uniform in linked shaders must have the same precision
// @feature: GLSL300.UNIFORM
// @feature: GLSL300.PRECISION.QUALIFIERS
uniform highp float u_scale;
in vec4 a_position;
void main() { gl_Position = a_position * u_scale; }
