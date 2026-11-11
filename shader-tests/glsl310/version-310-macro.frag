#version 310 es
// @expect: compile-pass
// @spec: GLSL ES 3.10 §3.3 "The directive "#version 310 es" is required in any shader that uses version 3.10"; §3.4 __VERSION__ is 310
// @feature: GLSL310.VERSION
#if __VERSION__ != 310 || GL_ES != 1
#error "__VERSION__ must be 310 and GL_ES must be 1"
#endif
precision mediump float;
out vec4 o;
void main() { o = vec4(1.0); }
