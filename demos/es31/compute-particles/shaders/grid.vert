#version 310 es
precision highp float;

layout(location = 0) in vec3 a_pos;  // y = 0 ground lines; a_pos.y carries 1 for the major lines
uniform mat4 u_view;
uniform mat4 u_proj;
out vec2 v_xz;
out float v_major;

void main()
{
    v_xz = a_pos.xz;
    v_major = a_pos.y;
    gl_Position = u_proj * u_view * vec4(a_pos.x, 0.0, a_pos.z, 1.0);
}
