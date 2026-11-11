#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.3 a constant expression is "a global or local variable qualified as const (i.e., not including function parameters)"
// @feature: GLSL300.QUAL.CONST
precision mediump float;
out vec4 o;
float f(const float p) { const float c = p; return c; }
void main() { o = vec4(f(1.0)); }
