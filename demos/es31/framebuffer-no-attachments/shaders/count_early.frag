#version 310 es
// The same shader as count.frag with ONE added line: layout(early_fragment_tests) in;. The depth test
// (and stencil test, occlusion queries) now run BEFORE the shader (ES 3.2 sections 13.8.4 and 14.2.4), so an
// invocation exists only for fragments that passed the depth test, and its buffer writes can be trusted to
// describe exactly the visible fragments.
precision highp float;
precision highp int;

layout(early_fragment_tests) in;

layout(std430, binding = 0) buffer Hits
{
    uint total;
    uint min_x;
    uint min_y;
    uint max_x;
    uint max_y;
    uint cells[];
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
