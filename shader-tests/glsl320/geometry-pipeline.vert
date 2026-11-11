#version 320 es
// @expect: link-pass
// @spec: GLSL ES 3.20 §4.4.1.2 Geometry Shader Inputs, §4.4.2.2 Geometry Outputs, §8.16 EmitVertex/EndPrimitive
// @feature: GLSL320.VERSION
// @feature: GLSL320.GEOMETRY.LAYOUT
// @feature: GLSL320.GEOMETRY.BUILTINS
// @feature: GLSL320.INTERFACE_BLOCKS_IO
out VertexData { vec3 color; } v_out;
void main() { v_out.color = vec3(1.0); gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }
