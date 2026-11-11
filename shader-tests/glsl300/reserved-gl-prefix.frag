#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.8 Identifiers - identifiers starting with "gl_" are reserved
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
void main() { float gl_mine = 1.0; o = vec4(gl_mine); }
