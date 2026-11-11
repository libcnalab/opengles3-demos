#version 300 es
// Every object in the scene is the same unit cube [0,1]^3, placed by a model matrix that is only a
// translation and a positive, axis-aligned scale. Such a matrix never changes the direction of a face
// normal, so the per-face normal is passed through unchanged.
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_mvp;

out vec3 v_normal;

void main()
{
    v_normal = a_normal;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
