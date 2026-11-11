#version 300 es
// @expect: link-pass
// @spec: GLSL ES 3.00 §4.3.6, §4.3.9 - flat integer varyings; smooth/centroid float varyings
// @feature: GLSL300.INTERP.SMOOTH_FLAT
// @feature: GLSL300.INTERP.CENTROID
// @feature: GLSL300.IO.IN_OUT
flat out uint v_id;
centroid out vec2 v_uv;
smooth out float v_t;
void main() { v_id = uint(gl_VertexID); v_uv = vec2(0.5); v_t = 1.0; gl_Position = vec4(0.0); }
