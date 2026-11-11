#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.9 Identifiers starting with "gl_" "may not be declared in a shader as either a variable or a function"
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
float gl_half(float x) { return x * 0.5; }
void main() { o = vec4(gl_half(1.0)); }
