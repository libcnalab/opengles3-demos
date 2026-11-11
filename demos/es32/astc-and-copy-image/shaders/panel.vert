#version 320 es
// A quad that fills the viewport (4-vertex triangle strip, no vertex buffer).
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));  // (0,0) (1,0) (0,1) (1,1)
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
