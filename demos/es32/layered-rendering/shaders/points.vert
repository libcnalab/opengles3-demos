#version 320 es
// Layer-content passes: the vertex shader only creates the point(s) the geometry shader expands.
precision highp float;

void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
