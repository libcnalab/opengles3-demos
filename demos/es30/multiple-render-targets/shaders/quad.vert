#version 300 es
// A quad covering the current viewport, generated from gl_VertexID (TRIANGLE_STRIP, 4 vertices,
// no vertex attributes). u_z is the NDC depth of the quad: the depth/stencil part of the demo
// moves it nearer or farther to see which fragments survive the depth test.
uniform float u_z;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    gl_Position = vec4(corner * 2.0 - 1.0, u_z, 1.0);
}
