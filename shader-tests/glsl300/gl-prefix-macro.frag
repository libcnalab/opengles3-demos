#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.5 "All macro names prefixed with "GL_" ... are also reserved, and defining such a name results in a compile-time error"
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
#define GL_MY_MACRO 1
void main() { o = vec4(float(GL_MY_MACRO)); }
