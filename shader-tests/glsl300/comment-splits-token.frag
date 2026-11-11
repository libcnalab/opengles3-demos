#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.6 "Comments are treated syntactically as a single space" - fl/**/oat is the two tokens fl and oat
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
void main() { fl/**/oat x = 1.0; o = vec4(x); }
