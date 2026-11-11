#version 300 es
// Writes the raw flat values into an RGBA32UI attachment so the CPU can compare them bit for bit:
//   R = gl_VertexID of the provoking vertex      G = flat uint palette index
//   B = bits of the flat float shade             A = bits of the SMOOTH shade (negative control)
precision highp float;
precision highp int;

flat in uint v_palette;
flat in float v_shade;
flat in int v_vertex_id;
in float v_smooth_shade;

layout(location = 0) out uvec4 o_ids;

void main()
{
    o_ids = uvec4(uint(v_vertex_id), v_palette, floatBitsToUint(v_shade), floatBitsToUint(v_smooth_shade));
}
