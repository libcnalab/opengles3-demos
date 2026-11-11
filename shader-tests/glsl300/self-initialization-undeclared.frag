#version 300 es
// @expect: compile-fail
// @spec: GLSL ES 3.00 §4.2.2 "int z = z; // error if z not previously defined" - the scope of a name starts after its initializer
// @feature: GLSL300.SCOPE
precision mediump float;
out vec4 o;
void main() { int z = z; o = vec4(float(z)); }
