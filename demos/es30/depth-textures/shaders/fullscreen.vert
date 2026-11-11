#version 300 es
// One oversized triangle that covers the whole viewport; no vertex buffer, only gl_VertexID.
// Every pass except the scene pass uses it; the fragment shaders work from gl_FragCoord.
void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
