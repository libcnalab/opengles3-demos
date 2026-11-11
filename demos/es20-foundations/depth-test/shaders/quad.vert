#version 300 es
// An axis-aligned quad drawn as a 4-vertex triangle strip. There are no vertex attributes: the
// corner comes from gl_VertexID, the rectangle and its depth from uniforms.
//
//   u_rect  = (x0, y0, x1, y1)  corners in normalized device coordinates
//   u_depth = (z0, slope)       NDC depth z_d = z0 + slope * x_d
//
// A non-zero slope makes the quad a tilted plane (needed to give glPolygonOffset's "factor"
// term something to scale). With slope = 0 the depth is the constant z0 at every vertex, so two
// quads that use the same z0 produce bit-identical depth values - the EQUAL tests rely on that.
uniform vec4 u_rect;
uniform vec2 u_depth;

void main()
{
    // gl_VertexID 0..3 -> (0,0) (1,0) (0,1) (1,1): a counter-clockwise strip.
    vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    vec2 p = mix(u_rect.xy, u_rect.zw, corner);
    gl_Position = vec4(p, u_depth.x + u_depth.y * p.x, 1.0);
}
