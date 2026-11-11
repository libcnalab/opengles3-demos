#version 300 es
// Main pass, fragment stage: everything the learning path built, in one place.
//
//   albedo     procedural texture (unit 0, trilinear mipmaps) * material colour * instance tint
//   normal     tangent-space normal map (unit 1) through the TBN frame        [key 2]
//   sun        Blinn-Phong (or Phong) directional light                       [key 6]
//   shadow     sampler2DShadow (unit 2): 3x3 taps of hardware PCF             [key 1]
//   ambient    hemisphere light: ground colour below, sky colour above
//   fog        exponential, towards the sky's horizon colour                  [key 3]
//   wire       barycentric edge overlay (core ES has no polygon mode)         [key 4]
//
// Lighting happens in linear space; the result is gamma-encoded (1/2.2) because the default
// framebuffer is not sRGB. The wire overlay is mixed in AFTER the encoding: it is a debug colour.
//
// GLSL ES 3.00 4.5.4: the fragment language has no default float precision, and samplers default
// to lowp (sampler2DShadow has no default at all), so all three are declared.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DShadow;

in vec3 v_world;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec2 v_uv;
in vec3 v_bary;
in vec3 v_tint;
in vec4 v_light;

uniform sampler2D u_albedo_map;        // unit 0
uniform sampler2D u_normal_map;        // unit 1
uniform sampler2DShadow u_shadow_map;  // unit 2, TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE, LEQUAL

// material
uniform vec3 u_albedo;
uniform int u_textured;        // 0: plain u_albedo
uniform int u_uv_mode;         // 0: mesh texture coordinates, 1: world x/z divided by u_ground_tile
uniform float u_ground_tile;
uniform int u_has_normal_map;
uniform float u_specular;
uniform float u_shininess;

// technique toggles
uniform int u_shadows;
uniform int u_normal_mapping;
uniform int u_blinn;           // 1 Blinn-Phong (N.H), 0 Phong (R.V)
uniform int u_wire;
uniform float u_fog_density;   // 0 = no fog

// frame
uniform vec3 u_eye;
uniform vec3 u_sun_dir;        // unit vector TOWARDS the sun
uniform vec3 u_sun_color;
uniform vec3 u_sky_ambient;
uniform vec3 u_ground_ambient;
uniform vec3 u_fog_color;      // linear; equals the sky colour at the horizon
uniform vec3 u_wire_color;
uniform float u_shadow_texel;  // 1 / shadow map size
uniform float u_shadow_bias;   // receiver bias in window-z units

layout(location = 0) out vec4 o_color;

// 1 = lit, 0 = shadowed, between = penumbra (es30-shadow-mapping).
float shadow_factor()
{
    vec3 sc = (v_light.xyz / v_light.w) * 0.5 + 0.5;  // NDC -> (s, t, window depth)
    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0)
        return 1.0;  // outside the sun's frustum nothing was recorded: lit
    float reference = sc.z - u_shadow_bias;
    float sum = 0.0;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i)
            // GLSL ES 3.00 8.8: the third component is Dref; the result is the LEQUAL comparison,
            // filtered over the 2x2 nearest texels because the filter is LINEAR (ES 3.0 3.8.15).
            sum += texture(u_shadow_map, vec3(sc.xy + vec2(float(i), float(j)) * u_shadow_texel, reference));
    return sum / 9.0;
}

void main()
{
    // Every implicit-derivative operation (the two texture() lookups and fwidth) happens here, in
    // uniform control flow, before anything depends on per-fragment branches.
    vec2 uv = (u_uv_mode == 1) ? v_world.xz / u_ground_tile : v_uv;
    vec3 tex_albedo = texture(u_albedo_map, uv).rgb;
    vec3 n_ts = texture(u_normal_map, uv).xyz * 2.0 - 1.0;
    // Barycentric coordinate i is 0 on edge i and grows linearly across the triangle, so b_i / fwidth(b_i) is (at most)
    // the distance to edge i in pixels; fwidth(b_i) is about sqrt(2) / (triangle height over edge i in pixels).
    vec3 bary_rate = fwidth(v_bary);
    vec3 bary_px = v_bary / max(bary_rate, vec3(1e-6));
    float edge_px = min(min(bary_px.x, bary_px.y), bary_px.z);
    float edge_rate = max(max(bary_rate.x, bary_rate.y), bary_rate.z);

    vec3 albedo = u_albedo * v_tint * ((u_textured != 0) ? tex_albedo : vec3(1.0));

    vec3 n_geo = normalize(v_normal);
    vec3 n = n_geo;
    if (u_has_normal_map != 0 && u_normal_mapping != 0)
        n = normalize(mat3(normalize(v_tangent), normalize(v_bitangent), n_geo) * n_ts);

    vec3 L = u_sun_dir;
    vec3 V = normalize(u_eye - v_world);
    float sf = (u_shadows != 0) ? shadow_factor() : 1.0;
    // A face turned away from the sun (by its GEOMETRIC normal) gets no direct light at all, whatever the
    // normal map says: it is in its own shadow.
    float lit = dot(n_geo, L) > 0.0 ? sf : 0.0;

    float diffuse = max(dot(n, L), 0.0);
    float s = (u_blinn != 0) ? max(dot(n, normalize(L + V)), 0.0)   // Blinn-Phong: half vector
                             : max(dot(reflect(-L, n), V), 0.0);    // Phong: mirror direction
    float spec = diffuse > 0.0 ? pow(s, u_shininess) : 0.0;
    vec3 ambient = mix(u_ground_ambient, u_sky_ambient, 0.5 * n.y + 0.5);

    vec3 color = albedo * (ambient + u_sun_color * (diffuse * lit)) + u_sun_color * (u_specular * spec * lit);

    float fog = exp(-u_fog_density * length(v_world - u_eye));
    color = mix(u_fog_color, color, fog);

    vec3 display = pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2));
    // Lines fade out on triangles only a few pixels tall (distant ground, sphere poles), where they would merge into a
    // solid colour.
    float wire = (u_wire != 0) ? (1.0 - smoothstep(0.75, 1.5, edge_px)) * (1.0 - smoothstep(0.12, 0.3, edge_rate)) : 0.0;
    o_color = vec4(mix(display, u_wire_color, wire), 1.0);
}
