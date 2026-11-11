#version 300 es
// No vertex attributes at all: the three corners of a triangle that covers the whole viewport
// are derived from gl_VertexID (a GLSL ES 3.00 built-in). Drawing with no enabled attribute
// arrays is legal in OpenGL ES 3.0.
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); // (0,0) (2,0) (0,2)
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);                         // covers [-1,1]^2
}
