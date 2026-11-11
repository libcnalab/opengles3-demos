// GLSL ES 1.00 allows comments and white space before #version (unlike 3.00, see
/* glsl300/version-not-first-line.frag). */

#version 100
// @expect: compile-pass
// @spec: GLSL ES 1.00 §3.4 "The #version directive must occur in a compilation unit before anything else, except for comments and white space"
// @feature: GLSL100.VERSION
attribute vec4 a_position;
void main() { gl_Position = a_position; }
