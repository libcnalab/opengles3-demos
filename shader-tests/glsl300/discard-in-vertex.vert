#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §6.4 Jumps - discard is only allowed in fragment shaders
// @feature: GLSL300.DISCARD
void main() { gl_Position = vec4(0.0); discard; }
