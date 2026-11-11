#version 310 es
// One triangle that covers the whole viewport (gl_VertexID 0, 1, 2 -> (-1,-1), (3,-1), (-1,3)).
// There are no vertex attributes: every fragment of the viewport is one "query" of the gather demo.
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
