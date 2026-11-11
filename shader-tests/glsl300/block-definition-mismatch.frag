#version 300 es
precision mediump float;
layout(std140) uniform Shared { vec4 b; vec4 a; };
out vec4 o;
void main() { o = a - b; }
