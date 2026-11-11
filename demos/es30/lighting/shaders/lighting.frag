#version 300 es
// Per-fragment ("Phong shading") lighting. Everything is evaluated in world space:
//
//   color = emissive + ambient * albedo
//         + sum over lights of  radiance * ( albedo * max(N.L, 0)  +  specular * pow(max(S, 0), shininess) )
//
// with S = R.V for Phong (R = reflect(-L, N)) and S = N.H for Blinn-Phong (H = normalize(L + V)),
// radiance = light color * distance attenuation * spot-cone falloff, and the specular term applied
// only where the surface faces the light (N.L > 0). The result is gamma-encoded (1/2.2) at the end.
//
// The fragment language has no default float precision; highp (mandatory in GLSL ES 3.00 fragment
// shaders) is declared because the specular exponent and the world-space positions need the range.
precision highp float;
precision highp int;

in vec3 v_world_pos;
in vec3 v_normal;

layout(location = 0) out vec4 o_color;

struct Material
{
    vec3 albedo;      // diffuse AND ambient reflectance (tile colour A when checker > 0)
    vec3 albedo2;     // tile colour B
    float checker;    // tile edge length in world units on the xz plane; 0 = uniform colour
    vec3 specular;    // specular reflectance
    float shininess;  // specular exponent
    vec3 emissive;    // light the surface emits by itself (lamps)
    int model;        // 0 = Phong, 1 = Blinn-Phong
};

struct DirLight   { vec3 direction; vec3 color; };                   // direction = the way the light travels
struct PointLight { vec3 position; vec3 color; vec3 attenuation; };  // attenuation = (constant, linear, quadratic)
struct SpotLight
{
    vec3 position;
    vec3 direction;   // unit vector, the way the cone points
    vec3 color;
    vec3 attenuation;
    float cos_inner;  // cosine of the inner cone half-angle: full intensity inside
    float cos_outer;  // cosine of the outer cone half-angle: no light outside (cos_outer < cos_inner)
};

uniform Material u_material;
uniform vec3 u_eye;      // camera position in world space
uniform vec3 u_ambient;  // colour of the constant ambient light
uniform DirLight u_dir;
uniform PointLight u_point;
uniform SpotLight u_spot;

// Reflection of one light that arrives from the unit direction L (surface -> light).
vec3 reflect_light(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo)
{
    float n_dot_l = dot(N, L);
    if (n_dot_l <= 0.0)
        return vec3(0.0);  // facing away from the light: neither diffuse nor specular
    float s;
    if (u_material.model == 0)
    {
        vec3 R = reflect(-L, N);  // mirror direction of the incoming light
        s = max(dot(R, V), 0.0);
    }
    else
    {
        vec3 H = normalize(L + V);  // half vector between light and viewer
        s = max(dot(N, H), 0.0);
    }
    float spec = pow(s, u_material.shininess);
    return radiance * (albedo * n_dot_l + u_material.specular * spec);
}

void main()
{
    // Interpolated normals are shorter than 1 inside a triangle: renormalize per fragment.
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_eye - v_world_pos);

    vec3 albedo = u_material.albedo;
    if (u_material.checker > 0.0)
    {
        vec2 cell = floor(v_world_pos.xz / u_material.checker);
        albedo = mix(u_material.albedo, u_material.albedo2, mod(cell.x + cell.y, 2.0));
    }

    vec3 color = u_material.emissive + u_ambient * albedo;

    // Directional light: parallel rays, no attenuation.
    color += reflect_light(N, V, normalize(-u_dir.direction), u_dir.color, albedo);

    // Point light: attenuation 1 / (kc + kl d + kq d^2).
    {
        vec3 to_light = u_point.position - v_world_pos;
        float d = length(to_light);
        float atten = 1.0 / (u_point.attenuation.x + u_point.attenuation.y * d + u_point.attenuation.z * d * d);
        color += reflect_light(N, V, to_light / d, u_point.color * atten, albedo);
    }

    // Spot light: a point light times a smooth cone falloff between the outer and the inner angle.
    {
        vec3 to_light = u_spot.position - v_world_pos;
        float d = length(to_light);
        vec3 L = to_light / d;
        float atten = 1.0 / (u_spot.attenuation.x + u_spot.attenuation.y * d + u_spot.attenuation.z * d * d);
        float cos_theta = dot(-L, u_spot.direction);  // angle between the cone axis and the ray light -> fragment
        float cone = smoothstep(u_spot.cos_outer, u_spot.cos_inner, cos_theta);  // 0 outside, 1 inside, smooth between
        color += reflect_light(N, V, L, u_spot.color * (atten * cone), albedo);
    }

    o_color = vec4(pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
