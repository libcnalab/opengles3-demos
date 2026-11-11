#version 310 es
// No vertex attributes: the corners of one triangle that covers the whole viewport come from gl_VertexID.
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
