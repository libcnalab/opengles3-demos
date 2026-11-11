#version 300 es
// One triangle that covers the whole viewport, generated from gl_VertexID (GLSL ES 3.00 7.1): no vertex buffer,
// no attributes. Vertices 0, 1, 2 are (-1,-1), (3,-1), (-1,3) in clip space.
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
