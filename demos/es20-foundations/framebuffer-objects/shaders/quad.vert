#version 300 es
// A rectangle generated from gl_VertexID (draw it with TRIANGLE_STRIP and 4 vertices; no vertex
// buffer or attribute is involved). u_rect = (x0, y0, x1, y1) in normalized device coordinates and
// u_z is the NDC depth of the whole rectangle, so one tiny shader serves every pass of the demo:
// full-target quads for texturing, and small rectangles at a chosen depth for the depth/stencil tests.
uniform vec4 u_rect;
uniform float u_z;
out vec2 v_uv;  // 0..1 across the rectangle
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, corner), u_z, 1.0);
    v_uv = corner;
}
