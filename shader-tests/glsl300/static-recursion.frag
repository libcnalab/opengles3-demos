#version 300 es
// @expect: build-fail
// @spec: GLSL ES 3.00 §6.1 Function Definitions - static and dynamic recursion is not allowed
// @feature: GLSL300.FUNCTIONS
precision mediump float;
out vec4 o;
float f(float x) { return x <= 0.0 ? 0.0 : f(x - 1.0); }
void main() { o = vec4(f(3.0)); }
