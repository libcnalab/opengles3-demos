#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.3.4 - fragment inputs that are or contain integers must be qualified flat
// @feature: GLSL300.INTERP.SMOOTH_FLAT
precision mediump float;
in highp int v_id;
out vec4 o;
void main() { o = vec4(float(v_id)); }
