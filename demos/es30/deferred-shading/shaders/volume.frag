#version 300 es
// Deferred lighting with light volumes: one fragment shader invocation per (light, covered pixel) pair. Each
// writes only that light's contribution; the application sums them with additive blending
// (glBlendFunc(ONE, ONE)) into the light buffer, on top of the ambient + emissive base.
precision highp float;
precision highp int;

const int MAX_LIGHTS = 64;

layout(std140) uniform Lights
{
    vec4 u_pos_radius[MAX_LIGHTS];
    vec4 u_color[MAX_LIGHTS];
};

uniform highp sampler2D u_albedo;
uniform highp sampler2D u_normal;
uniform highp sampler2D u_material;
uniform highp sampler2D u_depth;

uniform vec3 u_eye;
uniform vec3 u_ray_f;
uniform vec3 u_ray_r;
uniform vec3 u_ray_u;
uniform vec2 u_clip;
uniform vec2 u_size;

flat in int v_light;

out vec4 o_light;

vec3 reconstruct_position(vec2 frag, float depth)
{
    float n = u_clip.x;
    float f = u_clip.y;
    float ndc_z = depth * 2.0 - 1.0;
    float z = 2.0 * f * n / ((f + n) - ndc_z * (f - n));
    vec2 ndc = frag / u_size * 2.0 - 1.0;
    return u_eye + z * (u_ray_f + u_ray_r * ndc.x + u_ray_u * ndc.y);
}

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
    ivec2 pix = ivec2(gl_FragCoord.xy);
    float depth = texelFetch(u_depth, pix, 0).r;
    if (depth >= 1.0) discard;  // background: no surface to light

    vec4 a = texelFetch(u_albedo, pix, 0);
    vec3 n = normalize(texelFetch(u_normal, pix, 0).xyz * 2.0 - 1.0);
    vec4 m = texelFetch(u_material, pix, 0);

    vec3 p = reconstruct_position(gl_FragCoord.xy, depth);
    vec3 v = normalize(u_eye - p);

    vec3 radiance = light_contribution(v_light, p, n, v, a.rgb, a.a, m.r * 255.0);
    o_light = vec4(radiance * 0.5, 0.0);  // alpha 0: the coverage mark of the base pass must stay 1
}
