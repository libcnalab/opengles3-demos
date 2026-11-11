#version 320 es
// Cell C: shared by the plain fill pass (vertex + fragment shader) and the silhouette pass (vertex +
// geometry + fragment shader). The output block VS_OUT is matched BY NAME with the block of the same name in the
// next stage (the fragment shader of the fill program, the geometry shader of the silhouette program).
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

layout(location = 0) uniform mat4 u_viewproj;

out VS_OUT
{
    vec3 world;
    vec3 normal;
} vs_out;

void main()
{
    vs_out.world = a_position;
    vs_out.normal = a_normal;
    gl_Position = u_viewproj * vec4(a_position, 1.0);
}
