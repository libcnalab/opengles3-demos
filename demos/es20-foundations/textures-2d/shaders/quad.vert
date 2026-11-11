#version 300 es
// A quad covering the current viewport, generated from gl_VertexID (TRIANGLE_STRIP, 4 vertices).
// Texture coordinates span [offset, offset + scale] across the quad, so a cell can show a
// texture once (scale 1) or sample outside [0,1] to demonstrate the wrap modes.
uniform vec2 u_uv_scale;
uniform vec2 u_uv_offset;
out vec2 v_uv;
void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_uv = corner * u_uv_scale + u_uv_offset;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
