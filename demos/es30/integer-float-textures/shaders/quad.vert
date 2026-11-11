#version 300 es
// A quad covering the current viewport, generated from gl_VertexID (TRIANGLE_STRIP, 4 vertices,
// no vertex attributes). Every fragment shader of this demo derives what it needs from
// gl_FragCoord, so nothing is passed down.
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
