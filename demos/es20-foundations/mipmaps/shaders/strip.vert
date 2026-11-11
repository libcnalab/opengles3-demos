#version 300 es
// A quad covering the whole viewport (TRIANGLE_STRIP, 4 vertices, no vertex buffer).
//
// Only the texture coordinate s varies, linearly with the window x coordinate; t is constant. The
// derivatives the GL needs for the level of detail are therefore
//     du/dx = width * ds/dx,   du/dy = 0,   dv/dx = dv/dy = 0,
// and the scale factor of ES 3.0 equation 3.20 is exactly rho = |du/dx|, however an implementation
// chooses to approximate it (section 3.8.10.1 allows max{mu, mv} <= f <= mu + mv, and here
// mv = 0). That makes the level of detail of every strip a number the demo can state in advance.
uniform float u_s0;       // s at the left edge of the viewport
uniform float u_s_range;  // s at the right edge minus s at the left edge
out float v_s;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_s = u_s0 + corner.x * u_s_range;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
