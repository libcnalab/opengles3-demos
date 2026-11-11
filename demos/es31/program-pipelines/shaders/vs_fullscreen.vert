#version 310 es
// A separable vertex program with NO user-defined outputs. It pairs with fragment programs that
// declare no inputs (the uniform zoo, the solid fill): an empty interface matches exactly.
void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
