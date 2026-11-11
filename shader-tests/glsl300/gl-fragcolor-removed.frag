#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §7.2 - gl_FragColor/gl_FragData do not exist; declare an out variable
// @feature: GLSL300.FRAG.OUTPUT
precision mediump float;
void main() { gl_FragColor = vec4(1.0); }
