#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §5.8 "swizzles with repeated fields ... cannot be l-values"
// @feature: GLSL300.OPERATORS
precision mediump float;
out vec4 o;
void main() { vec4 v = vec4(0.0); v.xx = vec2(1.0, 2.0); o = v; }
