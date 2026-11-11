#version 300 es
// A single point (GL_POINTS) at u_center, u_point_size pixels wide. Points are not polygons, so
// they always use the FRONT set of stencil state, whatever glFrontFace says (ES 3.0 §4.1.4).
uniform vec2 u_center;
uniform float u_point_size;

void main()
{
    gl_Position = vec4(u_center, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
