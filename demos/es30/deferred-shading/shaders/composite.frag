#version 300 es
// Final pass into the default framebuffer. Mode 0 turns the light buffer (linear radiance * 0.5, RGB10_A2) into
// a display color: radiance -> 1 - exp(-radiance * exposure) -> gamma 1/2.2. The other modes show one G-buffer
// plane, so every attachment of the geometry pass can be inspected on screen:
//
//   0 lit result          4 linearized depth
//   1 albedo              5 world position rebuilt from depth (fract(p + 0.25): a 1-unit grid)
//   2 normals             6 mosaic: albedo | normals over depth | lit result
//   3 material            7 raw 24-bit depth packed into RGB8 (used by the validation, not by a key)
precision highp float;
precision highp int;

uniform highp sampler2D u_albedo;
uniform highp sampler2D u_normal;
uniform highp sampler2D u_material;
uniform highp sampler2D u_depth;
uniform highp sampler2D u_light;

uniform int u_mode;
uniform float u_exposure;
uniform vec3 u_eye;
uniform vec3 u_ray_f;
uniform vec3 u_ray_r;
uniform vec3 u_ray_u;
uniform vec2 u_clip;
uniform vec2 u_size;

out vec4 o_color;

float linear_depth(float depth)
{
    float n = u_clip.x;
    float f = u_clip.y;
    return 2.0 * f * n / ((f + n) - (depth * 2.0 - 1.0) * (f - n));
}

vec3 reconstruct_position(vec2 frag, float depth)
{
    vec2 ndc = frag / u_size * 2.0 - 1.0;
    return u_eye + linear_depth(depth) * (u_ray_f + u_ray_r * ndc.x + u_ray_u * ndc.y);
}

vec3 sky(ivec2 pix)
{
    float t = (float(pix.y) + 0.5) / u_size.y;
    return mix(vec3(0.014, 0.018, 0.036), vec3(0.050, 0.070, 0.140), t);
}

vec3 tonemap(vec3 radiance)
{
    return pow(vec3(1.0) - exp(-radiance * u_exposure), vec3(1.0 / 2.2));
}

vec3 view_color(int mode, ivec2 pix)
{
    if (mode == 0)
    {
        vec4 l = texelFetch(u_light, pix, 0);
        if (l.a < 0.5) return sky(pix);  // coverage mark: 0 where no geometry was drawn
        return tonemap(l.rgb * 2.0);
    }

    float depth = texelFetch(u_depth, pix, 0).r;
    if (mode == 7)
    {
        uint v = min(uint(depth * 16777215.0 + 0.5), 16777215u);  // 24-bit integer; the clamp keeps depth 1.0 from rounding up to 2^24
        return vec3(float((v >> 16u) & 255u), float((v >> 8u) & 255u), float(v & 255u)) / 255.0;
    }
    if (depth >= 1.0) return vec3(0.0);

    if (mode == 1) return texelFetch(u_albedo, pix, 0).rgb;
    if (mode == 2) return texelFetch(u_normal, pix, 0).rgb;
    if (mode == 3)
    {
        float specular = texelFetch(u_albedo, pix, 0).a;
        vec2 m = texelFetch(u_material, pix, 0).rg;
        return vec3(specular, m.r * 2.0, m.g);  // specular strength, shininess / 127.5, emissive / 3
    }
    if (mode == 4) return vec3(exp(-linear_depth(depth) * 0.055));
    return fract(reconstruct_position(vec2(pix) + 0.5, depth) + 0.25);  // mode 5 (+ 0.25 keeps y = 0 away from the wrap)
}

void main()
{
    ivec2 pix = ivec2(gl_FragCoord.xy);
    vec3 c;
    if (u_mode == 6)
    {
        vec2 half_size = u_size * 0.5;
        bool right = gl_FragCoord.x >= half_size.x;
        bool top = gl_FragCoord.y >= half_size.y;
        // Each quadrant shows the full image subsampled 2:1 (nearest texel, no filtering).
        ivec2 sample_at = min(ivec2(mod(gl_FragCoord.xy, half_size)) * 2, ivec2(u_size) - 1);
        int quadrant_mode = top ? (right ? 2 : 1) : (right ? 0 : 4);
        c = view_color(quadrant_mode, sample_at);
    }
    else
    {
        c = view_color(u_mode, pix);
    }
    o_color = vec4(c, 1.0);
}
