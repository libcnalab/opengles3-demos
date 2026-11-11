#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §7.3 lists no gl_MaxImageUnits (added by GLSL ES 3.10 §7.2); undeclared identifiers are errors (§4 "All variables and functions must be declared before being used")
// @feature: GLSL300.BUILTIN_CONSTANTS
precision mediump float;
out vec4 o;
void main() { o = vec4(float(gl_MaxImageUnits)); }
