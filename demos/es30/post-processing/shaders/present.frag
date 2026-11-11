#version 300 es
// The last pass: turns the linear images produced by the earlier passes into what the window shows.
//
//   mode 0  raw       the linear scene as it is stored (too dark: no gamma correction)
//   mode 1  gamma     sRGB encode of the RGBA8 scene (the shader does the gamma correction)
//   mode 2  gamma-hw  sRGB encode of the SRGB8_ALPHA8 scene: the texture unit decodes the stored bytes to
//                     linear on every lookup (ES 3.0 section 3.8.16), the hardware encoded them on write
//   mode 3  blur      sRGB encode of the separable-Gaussian blurred scene
//   mode 4  sobel     the edge map (already a display value)
//   mode 5  bloom     sRGB encode of scene + strength * blurred bright-pass
//   mode 6  dof       sRGB encode of the scene mixed with its blur by a circle of confusion derived from the
//                     depth texture (depth of field)
//   mode 7  overview  seven vertical stripes showing modes 0..6 side by side (one pixel wide dividers)
//
// Blur, bloom and depth of field are computed in LINEAR light, where averaging is physically right
// (a 50% mix of black and white is 0.5 linear = 188 sRGB, not 128), and encoded once at the very end.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_scene;          // RGBA8 scene, linear values
uniform sampler2D u_scene_srgb;     // SRGB8_ALPHA8 scene: lookups return linear values
uniform sampler2D u_blur;           // blurred scene
uniform sampler2D u_edge;           // Sobel edge map
uniform sampler2D u_bloom;          // blurred bright-pass
uniform sampler2D u_depth;          // window-space depth of the RGBA8 scene (DEPTH_COMPONENT24 texture)

uniform int u_mode;
uniform int u_fb_srgb;              // 1 when the default framebuffer encodes writes to sRGB (then undo the encode here)
uniform float u_bloom_strength;
uniform float u_z_near;
uniform float u_z_far;
uniform float u_focus;              // eye-space distance that is in focus
uniform float u_sharp;              // half width of the in-focus zone
uniform float u_range;              // distance beyond the zone at which the blur is complete

layout(location = 0) out vec4 o_color;

vec3 srgb_encode(vec3 c)
{
    c = clamp(c, 0.0, 1.0);
    vec3 high = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(12.92 * c, high, step(0.0031308, c));
}

vec3 srgb_decode(vec3 c)
{
    vec3 high = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(c / 12.92, high, step(0.04045, c));
}

vec3 fetch(sampler2D s, ivec2 p)
{
    return texelFetch(s, p, 0).rgb;
}

// Eye-space distance along the view axis of a window-space depth value (default depth range [0,1]).
float linear_depth(float d)
{
    float z_ndc = 2.0 * d - 1.0;
    return 2.0 * u_z_near * u_z_far / (u_z_far + u_z_near - z_ndc * (u_z_far - u_z_near));
}

vec3 effect(int mode, ivec2 p)
{
    if (mode == 0) return fetch(u_scene, p);
    if (mode == 1) return srgb_encode(fetch(u_scene, p));
    if (mode == 2) return srgb_encode(fetch(u_scene_srgb, p));
    if (mode == 3) return srgb_encode(fetch(u_blur, p));
    if (mode == 4) return fetch(u_edge, p);
    if (mode == 5) return srgb_encode(fetch(u_scene, p) + u_bloom_strength * fetch(u_bloom, p));
    float z = linear_depth(texelFetch(u_depth, p, 0).r);
    float coc = clamp((abs(z - u_focus) - u_sharp) / u_range, 0.0, 1.0);
    return srgb_encode(mix(fetch(u_scene, p), fetch(u_blur, p), coc));
}

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 c;
    if (u_mode == 7)
    {
        int width = textureSize(u_scene, 0).x;
        int stripe = (p.x * 7) / width;
        bool divider = p.x > 0 && ((p.x - 1) * 7) / width != stripe;
        c = divider ? vec3(0.85) : effect(stripe, p);
    }
    else
    {
        c = effect(u_mode, p);
    }
    if (u_fb_srgb != 0) c = srgb_decode(c);
    o_color = vec4(c, 1.0);
}
