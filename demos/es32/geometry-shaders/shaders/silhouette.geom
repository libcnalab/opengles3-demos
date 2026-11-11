#version 320 es
// Cell C: silhouette edges from TRIANGLES_ADJACENCY. Vertices 0, 2, 4 are the triangle; vertices 1, 3, 5 are the
// vertices of the neighbouring triangles across the edges 0-2, 2-4 and 4-0 (ES 3.2 section 10.1.10). An edge is on
// the silhouette when this triangle faces the eye and the neighbour does not. Each such edge becomes a thin
// screen-space quad (one 4-vertex strip). Only front-facing triangles emit, so every edge is visited once.
precision highp float;

layout(triangles_adjacency) in;
layout(triangle_strip, max_vertices = 12) out;

layout(location = 0) uniform mat4 u_viewproj;
layout(location = 1) uniform vec3 u_eye;
layout(location = 2) uniform vec2 u_halfwidth;  // half the line width in NDC units
layout(location = 3) uniform float u_bias;      // clip-space z bias so that the outline wins the depth test

in VS_OUT
{
    vec3 world;
    vec3 normal;
} gs_in[];

out GS_OUT
{
    vec3 edge;   // the edge end point this corner belongs to (world space); captured by transform feedback
    float side;  // -1 or +1: which side of the edge this corner is on
} gs_out;

bool faces_eye(vec3 a, vec3 b, vec3 c)
{
    return dot(cross(b - a, c - a), u_eye - a) > 0.0;
}

void corner(vec3 p, vec4 clip, vec2 offset, float side)
{
    gs_out.edge = p;
    gs_out.side = side;
    gl_Position = vec4(clip.xy + offset * clip.w, clip.z - u_bias * clip.w, clip.w);
    EmitVertex();
}

void main()
{
    if (!faces_eye(gs_in[0].world, gs_in[2].world, gs_in[4].world)) return;

    for (int e = 0; e < 3; ++e)
    {
        vec3 a = gs_in[2 * e].world;
        vec3 b = gs_in[(2 * e + 2) % 6].world;
        vec3 c = gs_in[2 * e + 1].world;  // the neighbour triangle across a-b is (b, a, c)
        if (faces_eye(b, a, c)) continue;       // the neighbour faces the eye as well: not a silhouette edge

        vec4 ca = u_viewproj * vec4(a, 1.0);
        vec4 cb = u_viewproj * vec4(b, 1.0);
        vec2 d = normalize(cb.xy / cb.w - ca.xy / ca.w);
        vec2 off = vec2(-d.y, d.x) * u_halfwidth;
        corner(a, ca, -off, -1.0);
        corner(a, ca, +off, +1.0);
        corner(b, cb, -off, -1.0);
        corner(b, cb, +off, +1.0);
        EndPrimitive();
    }
}
