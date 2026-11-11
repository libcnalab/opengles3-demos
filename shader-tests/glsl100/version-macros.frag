#version 100
// @expect: compile-pass
// @spec: GLSL ES 1.00 §3.4 "__VERSION__ ... will substitute the decimal integer 100", "GL_ES will be defined and set to 1"
// @feature: GLSL100.VERSION
#if __VERSION__ != 100 || GL_ES != 1
#error "__VERSION__ must be 100 and GL_ES must be 1"
#endif
precision mediump float;
void main() { gl_FragColor = vec4(float(__VERSION__) / 100.0); }
