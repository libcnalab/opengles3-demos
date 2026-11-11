#version 300 es
// Instanced rendering driven by vertex ATTRIBUTES with divisors (ES 3.0 section 2.9, glVertexAttribDivisor).
//
//   location 0  a_corner  corner of the unit square; divisor 0: advances once per VERTEX
//   location 1  a_offset  lower-left corner of the instance's grid slot (NDC)
//   location 2  a_color   color of the instance
//
// What a_offset / a_color deliver depends only on the divisors stored in the bound VAO:
//   divisor 0  element = vertex index
//   divisor d  element = floor(gl_InstanceID / d)
// This shader never looks at gl_InstanceID - the GL does the indexing.
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec2 a_offset;
layout(location = 2) in vec3 a_color;

uniform float u_size;

out vec3 v_color;

void main()
{
    gl_Position = vec4(a_offset + a_corner * u_size, 0.0, 1.0);
    v_color = a_color;
}
