#version 300 es
// @expect: compile-pass
// @spec: GLSL ES 3.00 §4 (no implicit conversions) and §5.4.1 - conversions are written as constructors
// @feature: GLSL300.TYPES.IMPLICIT_CONVERSION
precision mediump float;
out vec4 o;
void main() { int i = 3; uint u = 4u; float f = float(i) + float(u) + float(true); o = vec4(f, float(int(2.5)), float(uint(i) + u), 1.0); }
