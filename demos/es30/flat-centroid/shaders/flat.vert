#version 300 es
// Low-poly scene vertex shader. Three of its outputs are flat: the integer palette index, the float
// shade and gl_VertexID. A flat output is not interpolated; every fragment of a triangle receives
// the value of the provoking vertex, which ES 3.0 Table 2.12 defines as the LAST vertex of the
// triangle (3i for independent triangles, i+2 for strips and fans, counting vertices from 1).
// v_smooth_shade carries the same per-vertex value without `flat`, as a negative control.

layout(location = 0) in vec2 a_position;  // NDC inside the panel viewport
layout(location = 1) in uint a_palette;   // fed with glVertexAttribIPointer (integer attribute)
layout(location = 2) in float a_shade;

flat out uint v_palette;      // integer outputs must be flat (GLSL ES 3.00 4.3.6)
flat out float v_shade;       // float, flat
flat out int v_vertex_id;     // gl_VertexID of the vertex that provided the flat values
out float v_smooth_shade;     // same data, smooth: interpolated across the triangle

void main()
{
    v_palette = a_palette;
    v_shade = a_shade;
    v_vertex_id = gl_VertexID;
    v_smooth_shade = a_shade;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
