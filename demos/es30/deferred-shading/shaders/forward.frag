#version 300 es
// Forward-rendered reference: the same scene, the same lights and the same lighting equations as the deferred
// path, evaluated per fragment from the interpolated world position and normal. It exists only so that the
// deferred result can be compared with an image that never passed through a G-buffer.
// The result is stored in the same format as the deferred light accumulation buffer: linear radiance * 0.5
// (RGB10_A2 holds [0, 1]); alpha 1 marks covered pixels.
precision highp float;
precision highp int;

const int MAX_LIGHTS = 64;

layout(std140) uniform Lights
{
    vec4 u_pos_radius[MAX_LIGHTS];  // xyz: world position, w: radius of influence
    vec4 u_color[MAX_LIGHTS];       // rgb: color * intensity
};

in vec3 v_position;
in vec3 v_normal;

uniform vec3 u_albedo;
uniform vec3 u_albedo2;
uniform float u_specular;
uniform float u_shininess;
uniform float u_emissive;
uniform int u_checker;
uniform vec3 u_eye;
uniform int u_light_count;

out vec4 o_light;

vec3 ambient_term(vec3 n)
{
    return mix(vec3(0.020, 0.020, 0.028), vec3(0.060, 0.070, 0.100), n.y * 0.5 + 0.5);
}

// Contribution of light i: windowed inverse-square-like attenuation (reaches exactly 0 at the radius, so a light
// volume of that radius loses nothing), Lambert diffuse and a Blinn-Phong highlight.
vec3 light_contribution(int i, vec3 p, vec3 n, vec3 v, vec3 albedo, float spec, float shininess)
{
    vec3 to_light = u_pos_radius[i].xyz - p;
    float r = u_pos_radius[i].w;
    float d2 = dot(to_light, to_light);
    float x = 1.0 - d2 / (r * r);
    if (x <= 0.0) return vec3(0.0);
    float att = x * x;
    vec3 l = to_light * inversesqrt(max(d2, 1e-12));
    float ndl = dot(n, l);
    if (ndl <= 0.0) return vec3(0.0);
    vec3 h = normalize(l + v);
    float s = spec * pow(max(dot(n, h), 0.0), shininess);
    return u_color[i].rgb * att * (albedo * ndl + vec3(s));
}

void main()
{
    vec3 n = normalize(v_normal);
    vec3 albedo = u_albedo;
    if (u_checker != 0)
    {
        float cell = floor(v_position.x * 0.5) + floor(v_position.z * 0.5);
        albedo = (n.y > 0.5 && mod(cell, 2.0) < 0.5) ? u_albedo : u_albedo2;
    }
    vec3 v = normalize(u_eye - v_position);

    vec3 radiance = albedo * ambient_term(n) + albedo * u_emissive;
    for (int i = 0; i < u_light_count; ++i)
        radiance += light_contribution(i, v_position, n, v, albedo, u_specular, u_shininess);

    o_light = vec4(radiance * 0.5, 1.0);
}
