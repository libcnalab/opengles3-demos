#version 310 es
// One triangle that covers the whole viewport (gl_VertexID 0, 1, 2 -> (-1,-1), (3,-1), (-1,3)).
// u_z is the normalized device depth of every fragment (the depth tests of the demo write one depth per sample).
uniform float u_z;

void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, u_z, 1.0);
}
