#version 310 es
// A fragment shader with NO output variable. Its only effect is on a shader storage buffer: every invocation
// counts itself (total), tracks the smallest and largest window position it saw, and increments the cell of its
// own window position (a per-pixel hit map). There is no color attachment to write to; the framebuffer may
// even have no attachment at all.
precision highp float;
precision highp int;

layout(std430, binding = 0) buffer Hits
{
    uint total;
    uint min_x;
    uint min_y;
    uint max_x;
    uint max_y;
    uint cells[];  // row-major hit map, u_stride cells per row (0 = do not record)
} hits;

uniform int u_stride;

void main()
{
    uvec2 p = uvec2(gl_FragCoord.xy);
    atomicAdd(hits.total, 1u);
    atomicMin(hits.min_x, p.x);
    atomicMin(hits.min_y, p.y);
    atomicMax(hits.max_x, p.x);
    atomicMax(hits.max_y, p.y);
    if (u_stride > 0)
    {
        uint index = p.y * uint(u_stride) + p.x;
        if (index < uint(hits.cells.length())) atomicAdd(hits.cells[index], 1u);
    }
}
