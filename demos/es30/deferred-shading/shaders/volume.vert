#version 300 es
// Light volume: one instance of a low-resolution sphere per light. gl_InstanceID selects the light in the uniform
// block, the sphere is scaled to the light's radius (times a factor that makes the polyhedron circumscribe the
// true sphere) and moved to the light. Only the pixels this geometry covers run the lighting fragment shader.
precision highp float;

const int MAX_LIGHTS = 64;

layout(std140) uniform Lights
{
    vec4 u_pos_radius[MAX_LIGHTS];
    vec4 u_color[MAX_LIGHTS];
};

layout(location = 0) in vec3 a_position;  // unit sphere

uniform mat4 u_view_proj;
uniform float u_volume_scale;

flat out int v_light;

void main()
{
    vec4 pr = u_pos_radius[gl_InstanceID];
    v_light = gl_InstanceID;
    gl_Position = u_view_proj * vec4(pr.xyz + a_position * (pr.w * u_volume_scale), 1.0);
}
