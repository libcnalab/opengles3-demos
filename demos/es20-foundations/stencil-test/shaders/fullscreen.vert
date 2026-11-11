#version 300 es
// One triangle that covers the whole viewport, derived from gl_VertexID (no vertex attributes).
//
// A single triangle has no internal diagonal. That matters here: stencil operations such as INCR,
// DECR and INVERT are applied once per fragment, so a pixel that two triangles of a quad both
// produced would be counted twice. (The edge rule makes that impossible for shared edges, but a
// one-triangle cover removes the question entirely.)
//
//   u_flip  = 0: counter-clockwise in window space (front-facing under the default glFrontFace)
//   u_flip  = 1: clockwise (back-facing under the default glFrontFace)
//   u_depth = NDC z of the whole triangle (window depth = u_depth / 2 + 1/2 with the default range)
uniform int u_flip;
uniform float u_depth;

void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    if (u_flip != 0) corner = corner.yx;                                        // (0,0) (0,2) (2,0)
    gl_Position = vec4(corner * 2.0 - 1.0, u_depth, 1.0);                       // covers [-1,1]^2
}
