#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §3.4 - #error causes the compilation to fail
// @feature: GLSL300.PREPROCESSOR
#error "intentional"
precision mediump float;
out vec4 o;
void main() { o = vec4(1.0); }
