#version 300 es
// @expect: link-pass
// @spec: GLSL ES 3.00 §4.3.7 Interface Blocks, §4.3.8.3 Uniform Block Layout Qualifiers
// @feature: GLSL300.INTERFACE_BLOCK
// @feature: GLSL300.LAYOUT.BLOCK
layout(std140) uniform Transforms { mat4 mvp; layout(row_major) mat3 normal; } u_xf;
layout(shared, column_major) uniform Material { vec4 color; } u_mat[2];
in vec4 a_position;
out vec4 v_color;
void main() { v_color = u_mat[1].color + vec4(u_xf.normal[0], 0.0); gl_Position = u_xf.mvp * a_position; }
