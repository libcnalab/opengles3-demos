#version 300 es
// A quad covering the current viewport (one cell of the result grid): TRIANGLE_STRIP, 4 vertices from gl_VertexID.
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
