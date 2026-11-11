#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §7.4 gl_DepthRange is a uniform; §4.3.5 "uniform variables are read-only"
// @feature: GLSL300.BUILTIN_UNIFORMS
precision mediump float;
out vec4 o;
void main() { gl_DepthRange.near = 0.5; o = vec4(gl_DepthRange.near); }
