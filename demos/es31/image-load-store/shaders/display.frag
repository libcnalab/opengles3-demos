#version 310 es
// Shows a texture produced by the compute shaders, one texel per u_scale x u_scale block of the cell
// that starts at window pixel u_origin (texelFetch only: no filtering). u_mode selects the sampler:
//   0  sampler2D (RGBA8) at level u_level
//   1  usampler2DArray (R32UI), layer u_layer
//   2  usampler3D (R32UI), slice u_layer
//   3  usampler2D (R32UI) as gray = value * u_value_scale.x + u_value_scale.y
// For modes 1 and 2 the value is  1000 * tag + index  (index 0..63 is the texel number inside the 8x8 image,
// tag identifies the layer / pass that wrote it); it is shown as 0.5 * gray(index) + 0.5 * tint(tag).
precision highp float;
precision highp int;

uniform highp sampler2D u_tex2d;       // unit 0
uniform highp usampler2DArray u_uarr;  // unit 1
uniform highp usampler3D u_u3d;        // unit 2
uniform highp usampler2D u_utex;       // unit 3
uniform int u_mode;
uniform int u_level;
uniform int u_layer;
uniform int u_scale;
uniform ivec2 u_origin;
uniform vec2 u_value_scale;
layout(location = 0) out vec4 o_color;

vec3 tagged_color(uint v)
{
    float idx = float(v % 1000u) / 63.0;
    uint tag = (v / 1000u) % 8u;
    vec3 tint = vec3(float(tag & 1u), float((tag >> 1) & 1u), float((tag >> 2) & 1u));
    return 0.5 * vec3(idx) + 0.5 * tint;
}

void main()
{
    ivec2 t = (ivec2(gl_FragCoord.xy) - u_origin) / u_scale;
    vec3 c;
    if (u_mode == 0)
        c = texelFetch(u_tex2d, t, u_level).rgb;
    else if (u_mode == 1)
        c = tagged_color(texelFetch(u_uarr, ivec3(t, u_layer), 0).x);
    else if (u_mode == 2)
        c = tagged_color(texelFetch(u_u3d, ivec3(t, u_layer), 0).x);
    else
        c = vec3(float(texelFetch(u_utex, t, 0).x) * u_value_scale.x + u_value_scale.y);
    o_color = vec4(c, 1.0);
}
