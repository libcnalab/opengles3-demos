#version 310 es
// @expect: compile-fail
// @spec: GLSL ES 3.20 §1.1 - geometry shaders are part of GLSL ES 3.20 (3.10 needs an extension, which is out of scope)
// @feature: GLSL320.VERSION
layout(points) in;
layout(points, max_vertices = 1) out;
void main() { gl_Position = gl_in[0].gl_Position; EmitVertex(); EndPrimitive(); }
