#version 300 es
// Deferred lighting as ONE full-screen pass: every pixel reads its surface from the G-buffer, rebuilds the world
// position from the depth texture and loops over u_light_count lights from the uniform block.
// With u_light_count = 0 it only writes the ambient + emissive base (the first pass of the light-volume path).
// Output: linear radiance * 0.5 (the RGB10_A2 light buffer holds [0, 1]); alpha 1 marks covered pixels.
precision highp float;
precision highp int;

const int MAX_LIGHTS = 64;

layout(std140) uniform Lights
{
    vec4 u_pos_radius[MAX_LIGHTS];
    vec4 u_color[MAX_LIGHTS];
};

// GLSL ES 3.00 gives samplers lowp by default; a lowp lookup of the 24-bit depth or the 10-bit normals could be
// reduced to a few bits, so every G-buffer sampler is declared highp.
uniform highp sampler2D u_albedo;    // unit 0: RGBA8
uniform highp sampler2D u_normal;    // unit 1: RGB10_A2
uniform highp sampler2D u_material;  // unit 2: RG8
uniform highp sampler2D u_depth;     // unit 3: DEPTH_COMPONENT24 (TEXTURE_COMPARE_MODE = NONE, NEAREST)

uniform vec3 u_eye;
uniform vec3 u_ray_f;   // camera forward
uniform vec3 u_ray_r;   // camera right * tan(fovy / 2) * aspect
uniform vec3 u_ray_u;   // camera up * tan(fovy / 2)
uniform vec2 u_clip;    // near, far
uniform vec2 u_size;    // viewport size in pixels
uniform int u_light_count;

out vec4 o_light;

// Window-space depth -> world position. The projection maps the eye distance z to NDC z = ((f+n) - 2fn/z) / (f-n),
// so z = 2fn / ((f+n) - ndc_z (f-n)); the pixel's view ray then scales to that distance along the camera axis.
vec3 reconstruct_position(vec2 frag, float depth)
{
    float n = u_clip.x;
    float f = u_clip.y;
    float ndc_z = depth * 2.0 - 1.0;
    float z = 2.0 * f * n / ((f + n) - ndc_z * (f - n));
    vec2 ndc = frag / u_size * 2.0 - 1.0;
    return u_eye + z * (u_ray_f + u_ray_r * ndc.x + u_ray_u * ndc.y);
}

vec3 ambient_term(vec3 n)
{
    return mix(vec3(0.020, 0.020, 0.028), vec3(0.060, 0.070, 0.100), n.y * 0.5 + 0.5);
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
    if (depth >= 1.0)
    {
        o_light = vec4(0.0);  // nothing was drawn here: the background
        return;
    }
    vec4 a = texelFetch(u_albedo, pix, 0);
    vec3 n = normalize(texelFetch(u_normal, pix, 0).xyz * 2.0 - 1.0);
    vec4 m = texelFetch(u_material, pix, 0);

    vec3 albedo = a.rgb;
    float spec = a.a;
    float shininess = m.r * 255.0;
    float emissive = m.g * 3.0;

    vec3 p = reconstruct_position(gl_FragCoord.xy, depth);
    vec3 v = normalize(u_eye - p);

    vec3 radiance = albedo * ambient_term(n) + albedo * emissive;
    for (int i = 0; i < u_light_count; ++i)
        radiance += light_contribution(i, p, n, v, albedo, spec, shininess);

    o_light = vec4(radiance * 0.5, 1.0);
}
