#version 310 es
// A triangle that covers the whole viewport, generated from gl_VertexID (3 vertices, no attributes).
// Only used in ES 3.1+ contexts, together with fetch_ms.frag (both shaders of a program have the same version).
void main()
{
    vec2 p = vec2(float((gl_VertexID & 1) * 4 - 1), float((gl_VertexID >> 1) * 4 - 1));
    gl_Position = vec4(p, 0.0, 1.0);
}
