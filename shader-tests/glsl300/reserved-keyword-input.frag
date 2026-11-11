#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.8 "The following are the keywords reserved for future use. Using them will result in an error" (input)
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
void main() { float input = 1.0; o = vec4(input); }
