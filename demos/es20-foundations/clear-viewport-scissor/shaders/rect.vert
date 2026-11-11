#version 300 es
// A "probe" vertex shader with no vertex attributes: the (up to) four clip-space corners of the
// shape come from a uniform array indexed by gl_VertexID, and the depth is one uniform.
// Drawn as a GL_TRIANGLE_STRIP of 4 vertices it is a quad (corner order: lower-left, lower-right,
// upper-left, upper-right); drawn with 3 vertices it is a single triangle.
//
// w is 1, so clip coordinates equal normalized device coordinates, and the window position is
// exactly what the viewport transform of ES 3.0 section 2.13.1 makes of u_vertex / u_z:
//   x_w = (w/2) * x_d + (x + w/2),   y_w = (h/2) * y_d + (y + h/2),   z_w = (f-n)/2 * z_d + (n+f)/2

uniform vec2 u_vertex[4];  // NDC x, y of each corner
uniform float u_z;         // NDC z shared by all corners

void main()
{
    gl_Position = vec4(u_vertex[gl_VertexID], u_z, 1.0);
}
