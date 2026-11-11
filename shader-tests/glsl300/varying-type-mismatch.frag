#version 300 es
precision mediump float;
in vec2 v_value;
out vec4 o;
void main() { o = vec4(v_value, 0.0, 1.0); }
