#version 300 es
// Instanced rendering with NO vertex attributes at all: everything is derived from the built-in
// inputs gl_VertexID (the vertex index within one instance) and gl_InstanceID (the instance number),
// GLSL ES 3.00 section 7.1. A 4-vertex TRIANGLE_STRIP forms a square from gl_VertexID; the instance
// number selects the grid slot, the size and the color.
uniform int u_columns;     // instances per row
uniform vec2 u_origin;     // center of slot 0 (NDC)
uniform float u_step;      // distance between slot centers (NDC)
uniform float u_size0;     // side length of instance 0
uniform float u_dsize;     // growth of the side length per instance

out vec3 v_color;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    vec2 slot = vec2(float(gl_InstanceID % u_columns), float(gl_InstanceID / u_columns));
    float size = u_size0 + u_dsize * float(gl_InstanceID);
    gl_Position = vec4(u_origin + slot * u_step + (corner - 0.5) * size, 0.0, 1.0);
    v_color = vec3(float(gl_InstanceID & 3) / 3.0, float((gl_InstanceID >> 2) & 3) / 3.0, 0.5);
}
