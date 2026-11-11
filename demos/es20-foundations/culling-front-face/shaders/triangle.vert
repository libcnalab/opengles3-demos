#version 300 es
// One generic attribute, no matrices: the application hands over positions that are already
// in normalized device coordinates (clip w = 1), and the viewport of each test cell maps them to
// window coordinates.
layout(location = 0) in vec2 a_position;

// (+-1, +-1) applied to x and y in the shader. Facing is decided from the signed area in
// WINDOW space (ES 3.0 section 3.6.1), so any transform with a negative determinant reverses the
// winding of a triangle:
//   ( 1,  1)  identity                              winding unchanged
//   (-1,  1)  mirror in x (determinant -1)          winding reversed
//   (-1, -1)  rotation by 180 degrees (determinant +1)  winding unchanged
// A negative glViewport width/height is NOT a way to mirror: it generates INVALID_VALUE.
uniform vec2 u_flip;

// Only used for GL_POINTS (the value is ignored for lines and triangles).
uniform float u_point_size;

void main()
{
    gl_Position = vec4(a_position * u_flip, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
