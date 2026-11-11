#version 300 es
// A rectangle generated from gl_VertexID (TRIANGLE_STRIP, 4 vertices, no vertex attributes).
// u_rect = (x0, y0, x1, y1) in normalized device coordinates, u_z is the NDC depth of the whole
// rectangle. The depth/stencil scenes of the demo are lists of such rectangles.
uniform vec4 u_rect;
uniform float u_z;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, corner), u_z, 1.0);
}
