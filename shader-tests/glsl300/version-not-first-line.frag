// A comment before the directive: in GLSL ES (unlike desktop GLSL) #version must be on line 1.
#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.4 "The #version directive must be present in the first line of a shader"
// @xfail: Mesa | Mesa 25.0 accepts comments before #version in ES shaders (GLSL ES 3.00 §3.4, 3.10/3.20 §3.3 forbid it)
// @feature: GLSL300.VERSION
precision mediump float;
out vec4 o;
void main() { o = vec4(1.0); }
