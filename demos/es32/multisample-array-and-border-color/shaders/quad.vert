#version 320 es
// A quad that fills the viewport (4-vertex triangle strip, no vertex buffer), with texture
// coordinates from u_uv.xy at the bottom-left corner to u_uv.zw at the top-right corner.
uniform highp vec4 u_uv;

out highp vec2 v_uv;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));  // (0,0) (1,0) (0,1) (1,1)
    v_uv = mix(u_uv.xy, u_uv.zw, corner);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
