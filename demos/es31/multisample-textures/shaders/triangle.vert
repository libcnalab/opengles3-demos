#version 310 es
// One triangle whose corners are given in pixels of a u_size x u_size target. Its edges cross pixels at arbitrary
// angles, so some samples of an edge pixel are covered and others are not.
uniform vec2 u_tri[3];
uniform float u_size;

void main()
{
    gl_Position = vec4(u_tri[gl_VertexID] / u_size * 2.0 - 1.0, 0.0, 1.0);
}
