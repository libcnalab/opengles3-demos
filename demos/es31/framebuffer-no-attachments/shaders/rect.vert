#version 310 es
// An axis-aligned rectangle given in normalized device coordinates (u_rect = x0, y0, x1, y1), drawn as a
// TRIANGLE_STRIP of 4 vertices generated from gl_VertexID. u_depth is the window-space depth in [0, 1]
// (default depth range), so NDC z = 2 * u_depth - 1.
uniform vec4 u_rect;
uniform float u_depth;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, corner), u_depth * 2.0 - 1.0, 1.0);
}
