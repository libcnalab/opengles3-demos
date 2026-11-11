#version 320 es
// HDR lighting: Lambert diffuse from one directional light and two point lights (1 / (1 + d^2) falloff) whose
// intensities are well above 1, plus an emissive term. The result is written to an RGBA16F target without any
// clamping, so values of 5, 10 or 12 are stored as they are. main.cpp repeats this formula on the CPU.
precision highp float;
precision highp int;

in vec3 v_position;
in vec3 v_normal;

uniform vec3 u_albedo;
uniform vec3 u_emission;
uniform int u_checker;          // 1: floor, albedo from a 1 x 1 world-unit checkerboard
uniform vec3 u_ambient;
uniform vec3 u_sun_dir;         // unit vector towards the sun
uniform vec3 u_sun_color;
uniform vec3 u_light_pos[2];
uniform vec3 u_light_color[2];

layout(location = 0) out vec4 o_color;

void main()
{
    vec3 n = normalize(v_normal);
    vec3 albedo = u_albedo;
    if (u_checker != 0)
    {
        int tile = (int(floor(v_position.x)) + int(floor(v_position.z))) & 1;
        albedo = vec3(tile == 0 ? 0.65 : 0.15);
    }

    vec3 lit = u_ambient + u_sun_color * max(dot(n, u_sun_dir), 0.0);
    for (int i = 0; i < 2; ++i)
    {
        vec3 to_light = u_light_pos[i] - v_position;
        float d2 = dot(to_light, to_light);
        lit += u_light_color[i] * max(dot(n, normalize(to_light)), 0.0) / (1.0 + d2);
    }
    o_color = vec4(albedo * lit + u_emission, 1.0);
}
