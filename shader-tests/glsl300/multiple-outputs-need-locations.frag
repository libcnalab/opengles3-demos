#version 300 es
// @expect: compile-fail
// @spec: ES 3.0 §3.9.2.3 "If there is more than one output variable, all output variables must be explicitly bound ... Missing or conflicting binding assignments will cause CompileShader to fail"; GLSL ES 3.00 §4.3.8.2
// @xfail: Mesa | Mesa 25.0 compiles and links multiple fragment outputs without locations, assigning them itself
// @feature: GLSL300.FRAG.MRT
// @feature: GLSL300.LAYOUT.LOCATION
precision mediump float;
out vec4 o_a;
out vec4 o_b;
void main() { o_a = vec4(1.0); o_b = vec4(0.0); }
