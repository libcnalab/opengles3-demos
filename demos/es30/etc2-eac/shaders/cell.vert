#version 300 es
// One textured quad per cell, generated from gl_VertexID (TRIANGLE_STRIP, 4 vertices, no attributes).
// u_rect is the quad in normalized device coordinates (x0, y0, x1, y1); u_uv is the texel
// rectangle it shows, in texture coordinates (u0, v0, u1, v1).
uniform vec4 u_rect;
uniform vec4 u_uv;
out vec2 v_uv;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_uv = mix(u_uv.xy, u_uv.zw, corner);
    gl_Position = vec4(mix(u_rect.xy, u_rect.zw, corner), 0.0, 1.0);
}
