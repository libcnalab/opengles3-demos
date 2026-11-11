#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §7.2 "a shader may assign values to either gl_FragColor or gl_FragData, but not both"
// @feature: GLSL100.ATTRIBUTE_VARYING
precision mediump float;
void main() { gl_FragColor = vec4(1.0); gl_FragData[0] = vec4(0.0); }
