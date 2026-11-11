#version 300 es
// A quad covering the current viewport (one cell of the contact sheet), generated from
// gl_VertexID (TRIANGLE_STRIP, 4 vertices, no vertex attributes). v_corner runs from (0,0) at the
// lower left to (1,1) at the upper right of the cell; the fragment shader turns it into texture
// coordinates.
out vec2 v_corner;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_corner = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
