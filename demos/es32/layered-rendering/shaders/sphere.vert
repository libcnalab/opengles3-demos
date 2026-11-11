#version 320 es
// A UV sphere of radius 1 centred on the origin, where the environment was captured. The vertex position is also its
// normal.
precision highp float;

layout(location = 0) in vec3 a_position;

layout(location = 0) uniform mat4 u_viewproj;

out vec3 v_position;

void main()
{
    v_position = a_position;
    gl_Position = u_viewproj * vec4(a_position, 1.0);
}
