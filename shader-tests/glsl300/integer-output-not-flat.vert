#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.6 - vertex shader outputs containing integers must be qualified flat
// @xfail: Mesa | Mesa 25.0 only enforces flat on integer fragment inputs, not on vertex outputs
// @feature: GLSL300.INTERP.SMOOTH_FLAT
out int v_id;
void main() { v_id = gl_VertexID; gl_Position = vec4(0.0); }
