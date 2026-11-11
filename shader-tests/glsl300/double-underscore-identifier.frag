#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §3.8 "identifiers containing two consecutive underscores (__) are reserved ... Defining such a name in a shader does not itself result in an error"; §3.5 same for macro names
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
#define MY__MACRO 2.0
float my__value = 0.25;
void main() { o = vec4(my__value * MY__MACRO); }
