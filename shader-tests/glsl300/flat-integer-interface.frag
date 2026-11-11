#version 300 es
precision mediump float;
flat in uint v_id;
centroid in vec2 v_uv;
smooth in float v_t;
out vec4 o;
void main() { o = vec4(float(v_id & 1u), v_uv, v_t); }
