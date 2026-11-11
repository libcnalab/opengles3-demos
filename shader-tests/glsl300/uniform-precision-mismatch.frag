#version 300 es
precision mediump float;
uniform mediump float u_scale;
out vec4 o;
void main() { o = vec4(u_scale); }
