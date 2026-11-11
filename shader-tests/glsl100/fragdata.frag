#version 100
precision mediump float;
varying vec2 v_uv;
void main() { gl_FragData[0] = vec4(v_uv, 0.0, 1.0); }
