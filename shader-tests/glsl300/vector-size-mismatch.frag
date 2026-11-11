#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.9 arithmetic on two vectors requires "vectors of the same size" - "All other cases are illegal"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { vec3 v = vec2(1.0) + vec3(1.0); o = vec4(v, 1.0); }
