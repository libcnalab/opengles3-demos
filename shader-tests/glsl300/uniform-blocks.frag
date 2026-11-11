#version 300 es
precision mediump float;
layout(std140) uniform Transforms { mat4 mvp; layout(row_major) mat3 normal; } u_xf;
in vec4 v_color;
out vec4 o;
void main() { o = v_color * u_xf.mvp[0][0]; }
