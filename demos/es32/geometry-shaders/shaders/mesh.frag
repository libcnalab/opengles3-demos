#version 320 es
// Cell C fill pass: smooth Blinn-Phong on the interpolated vertex normal. The block VS_OUT arrives directly from
// the vertex shader (no geometry shader in this program).
precision highp float;

layout(location = 2) uniform vec3 u_eye;
layout(location = 3) uniform vec3 u_light;

in VS_OUT
{
    vec3 world;
    vec3 normal;
} fs_in;

out vec4 frag_color;

void main()
{
    vec3 albedo = vec3(0.10, 0.62, 0.72);
    vec3 n = normalize(fs_in.normal);
    vec3 v = normalize(u_eye - fs_in.world);
    vec3 h = normalize(u_light + v);
    float diff = max(dot(n, u_light), 0.0);
    float spec = pow(max(dot(n, h), 0.0), 24.0);
    frag_color = vec4(albedo * (0.18 + 0.82 * diff) + vec3(0.30 * spec), 1.0);
}
