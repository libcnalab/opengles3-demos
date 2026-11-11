#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.8 Keywords "cannot be used for any other purpose than that defined by this document"
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
void main() { float flat = 1.0; o = vec4(flat); }
