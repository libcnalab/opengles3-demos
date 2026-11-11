#version 320 es
precision mediump float;
in vec3 g_color;
out vec4 o;
void main() { o = vec4(g_color, 1.0); }
