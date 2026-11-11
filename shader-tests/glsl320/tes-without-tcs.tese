#version 320 es
layout(triangles) in;
void main() { gl_Position = gl_in[0].gl_Position; }
