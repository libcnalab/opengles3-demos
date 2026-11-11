#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.2 "Initializers for const declarations must be constant expressions" - a uniform is not one (§4.3.3)
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
uniform float u;
void main() { const float c = u; o = vec4(c); }
