#version 300 es
// One textured atlas tile, drawn into a small multisample renderbuffer. Each attribute is emitted twice:
// once with the default (pixel-centre) interpolation and once with `centroid out`.
//   a_uv              texture coordinate inside one 16x16 cell of a 48x48 atlas (inset by 1/8 texel)
//   a_edge_distance   signed distance in pixels from the vertex to each of the tile's four edge lines,
//                     positive inside. Interpolated, it is the distance of the evaluation point from
//                     each edge, so "min(d) < 0" means "evaluated outside the primitive".

layout(location = 0) in vec2 a_position;       // pixels inside the view
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_edge_distance;

uniform vec2 u_view_size;

out vec2 v_uv;
centroid out vec2 v_uv_centroid;
out vec4 v_dist;
centroid out vec4 v_dist_centroid;

void main()
{
    v_uv = a_uv;
    v_uv_centroid = a_uv;
    v_dist = a_edge_distance;
    v_dist_centroid = a_edge_distance;
    gl_Position = vec4(a_position / u_view_size * 2.0 - 1.0, 0.0, 1.0);
}
