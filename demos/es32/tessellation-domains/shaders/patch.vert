#version 320 es
// Vertex shader of the tessellation pipeline: it only hands the application's patch vertices to the tessellation
// control shader. Tessellation operates on PATCHES (ES 3.2 section 10.1.12); the vertex shader runs once per patch vertex.
layout(location = 0) in vec3 a_position;
layout(location = 1) in float a_tag;   // one number per patch vertex, passed through the whole pipeline

out float v_tag;

void main()
{
    gl_Position = vec4(a_position, 1.0);
    v_tag = a_tag;
}
