#version 300 es
// Pass 2: Blinn-Phong lighting of a directional light, multiplied by the shadow term from the
// shadow map. Everything outside the direct light (hemisphere ambient) is NOT shadowed, so a
// shadowed pixel shows exactly "albedo * ambient".
//
// GLSL ES 3.00 4.5.4: sampler2DShadow has NO default precision (sampler2D defaults to lowp), so the
// precision statement below is mandatory, and a lowp depth texture lookup would be useless anyway.
precision highp float;
precision highp sampler2DShadow;

in vec3 v_world;
in vec3 v_normal;
in vec4 v_light;

uniform sampler2DShadow u_shadow;  // depth texture with TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE
uniform float u_texel;             // 1 / shadow map size
uniform float u_bias;              // receiver depth bias in window-z units (0 = off)
uniform int u_pcf_radius;          // 0: one tap, 1: 3x3 taps (each tap is itself a hardware 2x2 PCF when LINEAR)

uniform vec3 u_eye;
uniform vec3 u_light_dir;          // unit vector pointing TOWARDS the light
uniform vec3 u_light_color;
uniform vec3 u_sky_ambient;
uniform vec3 u_ground_ambient;
uniform vec3 u_fog_color;
uniform float u_fog_density;

uniform vec3 u_albedo;
uniform vec3 u_albedo2;            // second checker color
uniform float u_checker;           // 1.0: ground checker board, 0.0: plain albedo
uniform float u_specular;
uniform float u_shininess;

out vec4 o_color;

// 1.0 = fully lit, 0.0 = fully in shadow, in between = penumbra.
float shadow_factor()
{
    vec3 sc = (v_light.xyz / v_light.w) * 0.5 + 0.5;  // NDC -> [0,1]: (s, t, depth)
    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0)
        return 1.0;  // outside the shadow map: nothing can have been recorded there, treat as lit

    float reference = sc.z - u_bias;
    float sum = 0.0;
    float count = 0.0;
    for (int j = -u_pcf_radius; j <= u_pcf_radius; ++j)
    {
        for (int i = -u_pcf_radius; i <= u_pcf_radius; ++i)
        {
            // GLSL ES 3.00 8.8: for a shadow sampler the last component of P is Dref. The result is
            // the comparison "Dref <= stored depth" (COMPARE_FUNC = LEQUAL), 0.0 or 1.0, or a
            // weighted mix of such results when the texture filter is LINEAR (ES 3.0 3.8.15).
            sum += texture(u_shadow, vec3(sc.xy + vec2(float(i), float(j)) * u_texel, reference));
            count += 1.0;
        }
    }
    return sum / count;
}

void main()
{
    vec3 n = normalize(v_normal);
    vec3 l = normalize(u_light_dir);
    vec3 v = normalize(u_eye - v_world);
    vec3 h = normalize(l + v);

    // Checker board; beyond 25 units it fades to its average, otherwise the far ground aliases into moire.
    float dist = length(v_world - u_eye);
    float checker = mod(floor(v_world.x) + floor(v_world.z), 2.0);
    checker = mix(0.5, checker, 1.0 - smoothstep(25.0, 70.0, dist));
    vec3 albedo = mix(u_albedo, u_albedo2, checker * u_checker);

    float diffuse = max(dot(n, l), 0.0);
    float spec = diffuse > 0.0 ? pow(max(dot(n, h), 0.0), u_shininess) : 0.0;
    vec3 ambient = mix(u_ground_ambient, u_sky_ambient, 0.5 * n.y + 0.5);

    float lit = shadow_factor();
    vec3 color = albedo * (ambient + u_light_color * (diffuse * lit)) + u_light_color * (u_specular * spec * lit);

    float fog = exp(-u_fog_density * dist);
    color = mix(u_fog_color, color, fog);

    o_color = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);  // output gamma (the default framebuffer is not sRGB)
}
