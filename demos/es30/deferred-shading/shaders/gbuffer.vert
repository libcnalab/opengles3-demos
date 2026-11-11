#version 300 es
// Geometry-pass vertex shader, shared by the G-buffer program and the forward reference program so that both
// rasterize bit-identical triangles (same code, same inputs -> same positions, same coverage, same depth).
//
//   a_position, a_normal  object-space attributes (interleaved, 24 bytes per vertex)
//   u_model               object -> world
//   u_normal_matrix       inverse transpose of the upper-left 3x3 of u_model; it keeps normals perpendicular
//                         to the surface when the model matrix scales non-uniformly (mat3 uniform)
//   u_view_proj           world -> clip
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform mat3 u_normal_matrix;

out vec3 v_position;  // world-space position
out vec3 v_normal;    // world-space normal (not unit length after interpolation)

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    v_position = world.xyz;
    v_normal = u_normal_matrix * a_normal;
    gl_Position = u_view_proj * world;
}
