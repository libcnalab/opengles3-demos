#version 300 es
// Ground plane and objects. Every model matrix is a rotation, a UNIFORM scale and a translation, so
// mat3(u_model) transforms normals correctly up to their length (renormalized in the fragment shader).
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_world;
out vec3 v_normal;

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world = world.xyz;
    v_normal = mat3(u_model) * a_normal;
    gl_Position = u_proj * u_view * world;
}
