#version 300 es
// Per-fragment Blinn-Phong lighting with a tangent-space normal map and a specular map.
//
//   unit 0  u_albedo    RGBA8  base colour
//   unit 1  u_normal    RGBA8  tangent-space normal, rgb = n * 0.5 + 0.5  (OpenGL convention: +Y = +v)
//   unit 2  u_specular  R8     specular mask (1 = shiny, 0 = matte)
//
// The perturbed normal is  N = normalize(T * n.x + B * n.y + N0 * n.z)  where (T, B, N0) are the
// interpolated world-space frame vectors from the vertex shader.
precision highp float;
precision highp int;
precision highp sampler2D;  // the default for samplers in a fragment shader is lowp

uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_specular;

uniform vec3 u_light_pos;     // world space point light
uniform vec3 u_light_color;
uniform vec3 u_eye;           // world space camera position
uniform float u_ambient;
uniform float u_shininess;
uniform int u_use_normal_map;  // 0: use the interpolated geometric normal ("flat" shading)
uniform int u_use_spec_map;    // 0: constant specular strength 1.0 everywhere
uniform int u_view;            // 0 lit, 1 albedo, 2 normal map texel, 3 specular map texel

in vec3 v_position;
in vec2 v_uv;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec3 v_normal;

layout(location = 0) out vec4 o_color;

void main()
{
    // All three lookups happen before any branch so that the implicit derivatives (level of
    // detail) are computed in uniform control flow.
    vec3 albedo = texture(u_albedo, v_uv).rgb;
    vec3 n_ts = texture(u_normal, v_uv).xyz * 2.0 - 1.0;
    float spec_texel = texture(u_specular, v_uv).r;

    if (u_view == 1) { o_color = vec4(albedo, 1.0); return; }
    if (u_view == 2) { o_color = vec4(n_ts * 0.5 + 0.5, 1.0); return; }
    if (u_view == 3) { o_color = vec4(vec3(spec_texel), 1.0); return; }

    if (u_use_normal_map == 0) n_ts = vec3(0.0, 0.0, 1.0);
    float spec_mask = (u_use_spec_map != 0) ? spec_texel : 1.0;

    vec3 n_geo = normalize(v_normal);
    mat3 tbn = mat3(v_tangent, v_bitangent, v_normal);  // columns T, B, N
    vec3 n = normalize(tbn * n_ts);

    vec3 to_light = u_light_pos - v_position;
    float dist = length(to_light);
    vec3 l = to_light / dist;
    vec3 v = normalize(u_eye - v_position);
    vec3 h = normalize(l + v);

    float attenuation = 1.0 / (1.0 + 0.12 * dist + 0.04 * dist * dist);
    // A bumped normal may face the light while the real surface does not: fade the light out
    // as the geometric normal turns away from it (self-shadowing of the bumps' back side).
    float facing = smoothstep(0.0, 0.2, dot(n_geo, l));

    float n_dot_l = max(dot(n, l), 0.0);
    float spec = (n_dot_l > 0.0) ? pow(max(dot(n, h), 0.0), u_shininess) : 0.0;
    float diffuse = n_dot_l * facing;
    spec *= facing;

    vec3 color = albedo * (u_ambient + u_light_color * diffuse * attenuation)
               + u_light_color * spec * spec_mask * attenuation;
    o_color = vec4(color, 1.0);
}
