#version 100
// @expect: link-pass
// @spec: GLSL ES 1.00 §4.3 Storage Qualifiers (attribute, varying); ES 3.0 contexts accept #version 100
// @feature: GLSL100.VERSION
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute vec4 a_position;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = a_position; }
