#version 320 es
// Draws vertices that were captured by transform feedback (clip-space gl_Position of the tessellation evaluation shader) as
// points. The point size of a tessellation or geometry shader is always 1.0 in core ES 3.2 (section 13.5: the size is only
// taken from a vertex shader), so large dots need this extra pass with a vertex shader.
layout(location = 0) in vec4 a_clip;

uniform float u_size;

void main()
{
    gl_Position = a_clip;
    gl_PointSize = u_size;
}
