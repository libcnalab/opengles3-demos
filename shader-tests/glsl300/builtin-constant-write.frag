#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §7.3 the built-in constants are declared const; §4.3.2 "It is an error to write to a const variable outside of its declaration"
// @feature: GLSL300.BUILTIN_CONSTANTS
precision mediump float;
out vec4 o;
void main() { gl_MaxDrawBuffers = 8; o = vec4(1.0); }
