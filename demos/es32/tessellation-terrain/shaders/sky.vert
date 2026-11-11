#version 320 es
// One triangle that covers the whole window, generated from gl_VertexID (no vertex buffer): the sky is a background pass.
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
