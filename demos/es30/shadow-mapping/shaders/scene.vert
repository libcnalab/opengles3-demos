#version 300 es
// Pass 2: the camera's view of the scene. Besides the usual world position and normal the vertex
// shader also computes the position in the light's clip space; the interpolated value is what
// the fragment shader looks up in the shadow map.
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat3 u_normal_matrix;  // inverse transpose of the model matrix's upper 3x3
uniform mat4 u_view_proj;
uniform mat4 u_light_vp;

out vec3 v_world;
out vec3 v_normal;
out vec4 v_light;  // clip-space position as seen from the light

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world = world.xyz;
    v_normal = u_normal_matrix * a_normal;
    v_light = u_light_vp * world;
    gl_Position = u_view_proj * world;
}
