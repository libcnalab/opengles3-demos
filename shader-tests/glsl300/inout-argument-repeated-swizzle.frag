#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.1.1 (out/inout need l-values) and §5.8 "swizzles with repeated fields ... cannot be l-values"
// @feature: GLSL300.PARAM_QUALIFIERS
precision mediump float;
out vec4 o;
void f(inout vec2 x) { x += vec2(1.0); }
void main() { vec4 v = vec4(0.0); f(v.xx); o = v; }
