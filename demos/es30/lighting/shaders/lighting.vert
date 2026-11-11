#version 300 es
// Per-fragment lighting, vertex stage: the vertex shader only moves data. It transforms the position
// to world space (model matrix) and to clip space (view and projection matrices), and rotates the
// object-space normal to world space with the NORMAL MATRIX. All lighting happens in lighting.frag.

layout(location = 0) in vec3 a_position;  // object space
layout(location = 1) in vec3 a_normal;    // object space, unit length

uniform mat4 u_model;          // object -> world
uniform mat4 u_view;           // world  -> eye
uniform mat4 u_proj;           // eye    -> clip
uniform mat3 u_normal_matrix;  // transpose(inverse(mat3(u_model))), computed on the CPU

out vec3 v_world_pos;
out vec3 v_normal;  // world space, NOT normalized: interpolation shortens it, the fragment shader renormalizes

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_normal = u_normal_matrix * a_normal;
    gl_Position = u_proj * u_view * world;
}
