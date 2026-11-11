#version 100
// @expect: compile-fail
// @spec: GLSL ES 1.00 §4.3.3 "The attribute qualifier can be used only with the data types float, vec2, vec3, vec4, mat2, mat3, and mat4"
// @feature: GLSL100.ATTRIBUTE_VARYING
attribute ivec2 a_index;
void main() { gl_Position = vec4(vec2(a_index), 0.0, 1.0); }
