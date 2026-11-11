#version 300 es
// A quad covering the current viewport, generated from gl_VertexID (TRIANGLE_STRIP, 4 vertices, no
// vertex attributes). v_uv runs from (0,0) at the lower left to (1,1) at the upper right of the
// viewport; at the centre of pixel (x, y) of a w x h viewport it is exactly ((x + 0.5) / w, (y + 0.5) / h).
out vec2 v_uv;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_uv = corner;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
