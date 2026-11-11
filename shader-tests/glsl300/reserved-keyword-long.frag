#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.8 "The following are the keywords reserved for future use. Using them will result in an error" (long)
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
void main() { int long = 1; o = vec4(float(long)); }
