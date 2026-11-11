#version 300 es
// One triangle that covers the whole viewport, from gl_VertexID alone (no vertex buffer):
// vertex 0 -> (-1,-1), 1 -> (3,-1), 2 -> (-1,3).
void main()
{
    vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
