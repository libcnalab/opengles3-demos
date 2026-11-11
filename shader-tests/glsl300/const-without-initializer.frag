#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.2 const variables "must be initialized when declared"
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
void main() { const float c; o = vec4(c); }
