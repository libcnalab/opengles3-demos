#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.9 Identifiers "Identifiers starting with "gl_" are reserved for use by OpenGL ES, and may not be declared in a shader as either a variable or a function"
// @feature: GLSL300.COMMENTS_TOKENS
precision mediump float;
out vec4 o;
void main() { float gl_mine = 1.0; o = vec4(gl_mine); }
