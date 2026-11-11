#version 300 es
// One triangle that covers the whole 128x64 probe target (vertices from gl_VertexID, no attributes); the scissor
// rectangle selects the 4x4 block of the current case.
//
// v_lin is affine in NDC and therefore, with w = 1, affine in window coordinates: with the viewport spanning the
// 128x64 target, d(ndc.x)/dx = 2/128 and d(ndc.y)/dy = 2/64, so
//   v_lin.x = 64 ndc.x            -> d/dx = 1,    d/dy = 0
//   v_lin.y = 32 ndc.y + 16 ndc.x -> d/dx = 0.25, d/dy = 1
out vec4 v_lin;

void main()
{
    vec2 ndc = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    v_lin = vec4(64.0 * ndc.x, 32.0 * ndc.y + 16.0 * ndc.x, 0.0, 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
