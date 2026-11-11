#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.4 "Function declarations (prototypes) cannot occur inside of functions; they must be at global scope"
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
float h(float v) { return v; }
void main() { float h(float v); o = vec4(h(1.0)); }
