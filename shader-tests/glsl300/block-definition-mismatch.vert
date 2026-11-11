#version 300 es
// @expect: link-fail
// @spec: GLSL ES 3.00 §4.3.7 - a block with the same name in linked shaders must match (members, types, order)
// @feature: GLSL300.INTERFACE_BLOCK
layout(std140) uniform Shared { vec4 a; vec4 b; };
void main() { gl_Position = a + b; }
