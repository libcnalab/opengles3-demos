#version 320 es
// A full-viewport rectangle from gl_VertexID (GL_TRIANGLE_STRIP, 4 vertices, no vertex attributes).
// The viewport selects the pixels it covers: corners (0,0) (1,0) (0,1) (1,1) -> NDC -1..+1.
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
