#version 300 es
// Gouraud shading for comparison: the SAME lighting equation as lighting.frag, but evaluated once per
// VERTEX; the rasterizer then interpolates the resulting colour across the triangle. Highlights and cone
// edges that fall between vertices are lost or smeared (compare the two coarse spheres).

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;

struct Material
{
    vec3 albedo;
    vec3 albedo2;  // unused here (no checker per vertex); declared so both programs share one uniform interface
    float checker;
    vec3 specular;
    float shininess;
    vec3 emissive;
    int model;
};

struct DirLight   { vec3 direction; vec3 color; };
struct PointLight { vec3 position; vec3 color; vec3 attenuation; };
struct SpotLight
{
    vec3 position;
    vec3 direction;
    vec3 color;
    vec3 attenuation;
    float cos_inner;
    float cos_outer;
};

uniform Material u_material;
uniform vec3 u_eye;
uniform vec3 u_ambient;
uniform DirLight u_dir;
uniform PointLight u_point;
uniform SpotLight u_spot;

out vec3 v_color;  // linear colour, interpolated; gouraud.frag gamma-encodes it

vec3 reflect_light(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo)
{
    float n_dot_l = dot(N, L);
    if (n_dot_l <= 0.0)
        return vec3(0.0);
    float s;
    if (u_material.model == 0)
    {
        vec3 R = reflect(-L, N);
        s = max(dot(R, V), 0.0);
    }
    else
    {
        vec3 H = normalize(L + V);
        s = max(dot(N, H), 0.0);
    }
    float spec = pow(s, u_material.shininess);
    return radiance * (albedo * n_dot_l + u_material.specular * spec);
}

void main()
{
    vec4 world = u_model * vec4(a_position, 1.0);
    vec3 P = world.xyz;
    vec3 N = normalize(u_normal_matrix * a_normal);
    vec3 V = normalize(u_eye - P);
    vec3 albedo = u_material.albedo;

    vec3 color = u_material.emissive + u_ambient * albedo;
    color += reflect_light(N, V, normalize(-u_dir.direction), u_dir.color, albedo);
    {
        vec3 to_light = u_point.position - P;
        float d = length(to_light);
        float atten = 1.0 / (u_point.attenuation.x + u_point.attenuation.y * d + u_point.attenuation.z * d * d);
        color += reflect_light(N, V, to_light / d, u_point.color * atten, albedo);
    }
    {
        vec3 to_light = u_spot.position - P;
        float d = length(to_light);
        vec3 L = to_light / d;
        float atten = 1.0 / (u_spot.attenuation.x + u_spot.attenuation.y * d + u_spot.attenuation.z * d * d);
        float cone = smoothstep(u_spot.cos_outer, u_spot.cos_inner, dot(-L, u_spot.direction));
        color += reflect_light(N, V, L, u_spot.color * (atten * cone), albedo);
    }

    v_color = color;
    gl_Position = u_proj * u_view * world;
}
