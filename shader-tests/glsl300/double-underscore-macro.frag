#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §3.5 (3.10/3.20 §3.4) "all macro names containing two consecutive underscores ( __ ) are reserved for use by underlying software layers. Defining such a name in a shader does not itself result in an error" (a warning is allowed)
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
#define MY__MACRO 2.0
void main() { o = vec4(0.25 * MY__MACRO); }
