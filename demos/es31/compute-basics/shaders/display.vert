#version 310 es
// Draws one square cell per compute invocation. The per-instance attributes are NOT CPU data: they are the
// records that builtins.comp wrote into a shader storage buffer, fetched here as an ARRAY_BUFFER
// (glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT) separates the two uses). Where a cell sits comes from the
// recorded gl_GlobalInvocationID, what color it has from the recorded gl_WorkGroupID / gl_NumWorkGroups
// (hue) and gl_LocalInvocationIndex (brightness).
layout(location = 0) in uvec4 a_num_groups;  // xyz = gl_NumWorkGroups, w = gl_LocalInvocationIndex
layout(location = 1) in uvec4 a_group_id;    // xyz = gl_WorkGroupID
layout(location = 2) in uvec4 a_global_id;   // xyz = gl_GlobalInvocationID

uniform ivec2 u_target;      // framebuffer size in pixels
uniform ivec2 u_origin;      // window position of cell column 0, row 0
uniform int u_cell;          // pixels per cell
uniform ivec4 u_layout;      // x: z slices per row of tiles, y: cells in x, z: cells in y, w: row length for 1-D grids (0 = none)
uniform uint u_local_count;  // invocations per work group (from glGetProgramiv(COMPUTE_WORK_GROUP_SIZE))

flat out vec3 v_color;

const uvec3 kPalette[12] = uvec3[12](
    uvec3(235u, 64u, 64u), uvec3(240u, 140u, 40u), uvec3(235u, 220u, 60u), uvec3(150u, 220u, 60u),
    uvec3(60u, 200u, 90u), uvec3(50u, 200u, 190u), uvec3(60u, 160u, 235u), uvec3(80u, 90u, 240u),
    uvec3(150u, 80u, 235u), uvec3(225u, 80u, 205u), uvec3(235u, 235u, 235u), uvec3(150u, 110u, 70u));

void main()
{
    ivec3 g = ivec3(a_global_id.xyz);
    int col;
    int row;
    if (u_layout.w > 0)
    {
        // 1-D dispatch: wrap the line so that each row is one work group.
        col = g.x % u_layout.w;
        row = g.x / u_layout.w;
    }
    else
    {
        // z slices are laid out as tiles, one empty cell apart.
        col = g.x + (g.z % u_layout.x) * (u_layout.y + 1);
        row = g.y + (g.z / u_layout.x) * (u_layout.z + 1);
    }

    // The quad covers u_cell - 1 pixels, which leaves a one pixel gap and keeps every edge on a pixel boundary.
    ivec2 corner = ivec2(gl_VertexID & 1, gl_VertexID >> 1);
    ivec2 pixel = u_origin + ivec2(col, row) * u_cell + corner * (u_cell - 1);
    gl_Position = vec4(2.0 * vec2(pixel) / vec2(u_target) - 1.0, 0.0, 1.0);

    uint group_flat = a_group_id.x + a_num_groups.x * (a_group_id.y + a_num_groups.y * a_group_id.z);
    uint brightness = 96u + (159u * a_num_groups.w) / max(u_local_count - 1u, 1u);
    v_color = vec3((kPalette[group_flat % 12u] * brightness) / 255u) / 255.0;
}
