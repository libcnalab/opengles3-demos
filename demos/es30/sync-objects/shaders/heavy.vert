#version 300 es
// One triangle that covers the whole viewport, built from gl_VertexID (no vertex attributes):
// ids 0, 1, 2 give the corners (-1,-1), (3,-1), (-1,3).
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
