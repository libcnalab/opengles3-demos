#version 300 es
// Environment mapping and exponential-squared fog.
//
//   u_material 0  ground: checkered albedo lit by the ambient sky light and the sun (Lambert)
//   u_material 1  mirror: tint * sky(reflect(I, N))                    (chrome sphere, gold box)
//   u_material 2  glass:  mix(tint * sky(refract(I, N, eta)), sky(reflect(I, N)), F)
//                 with the Fresnel-Schlick factor F = F0 + (1 - F0) (1 - cos theta)^5
//
// I is the unit vector from the eye to the surface point, N the unit normal. reflect() and refract()
// are the GLSL ES 3.00 section 8.5 built-ins. Only the environment (the cube map) is reflected and
// refracted, never the other objects or the ground: a static environment map.
//
// Fog (ES has no fixed-function fog): f = exp(-(density * d)^2) with d the eye distance, and
// colour = mix(fog colour, surface colour, f). The fog colour is the sky colour at the horizon, so
// distant ground dissolves into the haze band of the sky.
precision highp float;

uniform samplerCube u_sky;
uniform vec3 u_eye;
uniform int u_material;
uniform vec3 u_tint;
uniform float u_eta;  // ratio of indices of refraction n1 / n2 (air -> glass: 1 / 1.5)
uniform float u_f0;   // reflectance at normal incidence ((n1 - n2) / (n1 + n2))^2

uniform vec3 u_albedo_a;
uniform vec3 u_albedo_b;
uniform float u_tile;
uniform vec3 u_ambient;
uniform vec3 u_sun_dir;  // unit vector towards the sun
uniform vec3 u_sun_color;

uniform vec3 u_fog_color;
uniform float u_fog_density;

in vec3 v_world;
in vec3 v_normal;
out vec4 o_color;

void main()
{
    vec3 N = normalize(v_normal);
    vec3 I = normalize(v_world - u_eye);
    vec3 color;
    if (u_material == 0)
    {
        float parity = mod(floor(v_world.x / u_tile) + floor(v_world.z / u_tile), 2.0);
        vec3 albedo = parity < 0.5 ? u_albedo_a : u_albedo_b;
        color = albedo * (u_ambient + u_sun_color * max(dot(N, u_sun_dir), 0.0));
    }
    else
    {
        // u_material is a uniform, so these lookups are in dynamically uniform control flow and their
        // implicit derivatives (mipmap selection in the windowed configuration) are well defined.
        vec3 R = reflect(I, N);
        vec3 reflected = texture(u_sky, R).rgb;
        if (u_material == 1)
        {
            color = u_tint * reflected;
        }
        else
        {
            vec3 T = refract(I, N, u_eta);
            // refract() returns vec3(0) for total internal reflection (k < 0). That cannot happen for
            // air -> glass (eta < 1), only with the inverted ratio of the negative control; the light
            // is then reflected entirely. A zero direction must never reach texture().
            T = dot(T, T) == 0.0 ? R : T;
            vec3 refracted = texture(u_sky, T).rgb;
            float m = 1.0 - clamp(dot(-I, N), 0.0, 1.0);
            float F = u_f0 + (1.0 - u_f0) * (m * m) * (m * m) * m;
            color = mix(u_tint * refracted, reflected, F);
        }
    }
    float fd = u_fog_density * length(v_world - u_eye);
    float f = exp(-fd * fd);
    o_color = vec4(mix(u_fog_color, color, f), 1.0);
}
