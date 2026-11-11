#version 300 es
// Fragment stage of the scene: Blinn-Phong lighting in LINEAR space with one directional "moon" light, two
// orbiting point lights, an optional procedural checkerboard, exponential-squared fog toward the horizon
// color, and emissive lamp bulbs. The result is written as is; the render target decides how it is
// stored (RGBA8: clamped to [0,1] and rounded; SRGB8_ALPHA8: clamped, sRGB-encoded by the hardware, rounded).
precision highp float;
precision highp int;

#define NUM_POINT_LIGHTS 2

uniform int u_kind;                                 // 0 = lit solid color, 1 = checkerboard floor, 2 = emissive lamp
uniform vec3 u_albedo;                              // kind 0
uniform vec3 u_albedo_a;                            // kind 1: one checker color
uniform vec3 u_albedo_b;                            // kind 1: the other checker color
uniform vec3 u_emissive;                            // kind 2
uniform float u_specular;                           // specular strength
uniform float u_shininess;                          // Blinn-Phong exponent

uniform vec3 u_eye;
uniform vec3 u_ambient;
uniform vec3 u_dir_to_light;                        // unit vector from the surface toward the directional light
uniform vec3 u_dir_color;
uniform vec3 u_point_pos[NUM_POINT_LIGHTS];
uniform vec3 u_point_color[NUM_POINT_LIGHTS];
uniform vec3 u_fog_color;
uniform float u_fog_k;

in vec3 v_world;
in vec3 v_normal;

layout(location = 0) out vec4 o_color;

// Analytically filtered checkerboard with unit squares (derivative-based box filter), so that the far field
// fades to the average color instead of aliasing. Inside a square the result is exactly 0.0 or 1.0.
float checker(vec2 p)
{
    vec2 w = max(fwidth(p), vec2(1.0e-4));
    vec2 i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
    return 0.5 - 0.5 * i.x * i.y;
}

// Diffuse + Blinn-Phong specular of one light direction L (unit, pointing from the surface to the light).
vec3 light_term(vec3 L, vec3 N, vec3 V, vec3 albedo)
{
    float n_dot_l = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = n_dot_l > 0.0 ? pow(max(dot(N, H), 0.0), u_shininess) : 0.0;
    return albedo * n_dot_l + vec3(u_specular * spec);
}

void main()
{
    if (u_kind == 2)
    {
        o_color = vec4(u_emissive, 1.0);
        return;
    }

    vec3 N = normalize(v_normal);
    vec3 to_eye = u_eye - v_world;
    float dist = length(to_eye);
    vec3 V = to_eye / dist;

    vec3 albedo = u_albedo;
    if (u_kind == 1) albedo = mix(u_albedo_a, u_albedo_b, checker(v_world.xz));

    vec3 color = albedo * u_ambient + u_dir_color * light_term(u_dir_to_light, N, V, albedo);
    for (int i = 0; i < NUM_POINT_LIGHTS; ++i)
    {
        vec3 to_light = u_point_pos[i] - v_world;
        float d = length(to_light);
        float attenuation = 1.0 / (1.0 + 0.09 * d + 0.032 * d * d);
        color += u_point_color[i] * attenuation * light_term(to_light / d, N, V, albedo);
    }

    float fog = 1.0 - exp(-(u_fog_k * dist) * (u_fog_k * dist));
    o_color = vec4(mix(color, u_fog_color, fog), 1.0);
}
