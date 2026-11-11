#version 300 es
// Scene pass: positions are already in world space; one view-projection matrix per target
// (camera A, camera B, or one of the six cube faces).
layout(location = 0) in vec3 a_position;
uniform mat4 u_view_proj;

void main()
{
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
