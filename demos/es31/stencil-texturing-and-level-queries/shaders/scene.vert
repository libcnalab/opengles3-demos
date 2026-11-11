#version 310 es
// One axis-aligned rectangle of the depth/stencil scene, in texel units of the render target.
// u_rect = (x0, y0, x1, y1), u_size = edge of the (square) target, u_depth = window-space depth in [0, 1]
// (the default depth range [0, 1] maps NDC z = 2 * depth - 1 back to it). All values are multiples of
// 1/64, so every number below is exact in single precision and rectangle edges fall on pixel boundaries.
uniform vec4 u_rect;
uniform float u_size;
uniform float u_depth;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    vec2 texel = mix(u_rect.xy, u_rect.zw, corner);
    gl_Position = vec4(texel / u_size * 2.0 - 1.0, u_depth * 2.0 - 1.0, 1.0);
}
