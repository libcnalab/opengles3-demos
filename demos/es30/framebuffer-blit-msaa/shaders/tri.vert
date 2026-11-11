#version 300 es
// One triangle whose corners are given in pixels of a u_size x u_size target. Its edges cross
// pixels at arbitrary angles, which is what makes multisampling visible.
uniform vec2 u_tri[3];
uniform float u_size;

void main()
{
    gl_Position = vec4(u_tri[gl_VertexID] / u_size * 2.0 - 1.0, 0.0, 1.0);
}
