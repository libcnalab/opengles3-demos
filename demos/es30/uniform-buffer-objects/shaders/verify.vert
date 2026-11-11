#version 300 es
// No vertex attributes: the three corners of a triangle that covers the whole viewport come from
// gl_VertexID (GLSL ES 3.00 section 7.1). The verification program has no uniform block in this stage,
// so ES 3.0 section 2.12.6.2 reports every block as referenced by the fragment shader only.
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
