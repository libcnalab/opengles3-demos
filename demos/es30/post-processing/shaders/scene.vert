#version 300 es
// Vertex stage of the lit 3D scene: model -> world -> clip space. The normal matrix is the inverse
// transpose of the model's upper-left 3x3 (it keeps normals perpendicular to the surface under
// non-uniform scale); the lighting is evaluated per fragment in world space.
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_matrix;

out vec3 v_world;
out vec3 v_normal;

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world = world.xyz;
    v_normal = u_normal_matrix * a_normal;
    gl_Position = u_view_proj * world;
}
