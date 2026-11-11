#version 310 es
// Same interface as vs_pair_normal.vert, but the u coordinate is mirrored. Replacing the vertex
// stage of a pipeline with this program therefore moves the pattern from the left to the right.
layout(location = 0) uniform vec3 u_tint;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_tint;

void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = vec2(1.0 - corner.x, corner.y);
    v_tint = u_tint;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
