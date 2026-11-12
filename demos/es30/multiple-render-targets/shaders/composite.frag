#version 300 es
// Shows one of the four attachments in one quarter of the window, scaled to the quarter's size. texelFetch reads the texel
// (x, y) of level 0 directly (no filtering, no coordinate normalization), which is the way to look
// at integer textures: they are not texture-filterable (ES 3.0 Table 3.13).
//
// The integer attachments are turned into colors with fixed formulas that the CPU repeats:
//   panel 2 (RGBA32UI)  r = x/127   g = y/127   b = (low byte of the counter w)/255
//   panel 3 (RG32I)     r = (x*y)/16129   g = x/127 (low byte of INT_MIN + x)   b = 0.4 if negative
precision highp float;
precision highp int;

uniform highp sampler2D u_albedo;
uniform highp sampler2D u_data;
uniform highp usampler2D u_ids;
uniform highp isampler2D u_signed;
uniform int u_panel;
uniform ivec2 u_origin;      // window position of the lower left corner of the panel
uniform ivec2 u_panel_size;  // the panel's size in pixels (a quarter of the window, any size)

out vec4 o_color;

void main()
{
    // Integer texel mapping: pixel p of a panel n pixels wide shows texel p * 128 / n (integer division).
    // Every step is exact integer arithmetic, so the CPU model picks the very same texel at every window size.
    ivec2 tex_size = textureSize(u_albedo, 0);  // all four attachments are 128 x 128
    ivec2 t = (ivec2(gl_FragCoord.xy) - u_origin) * tex_size / u_panel_size;
    vec3 c;
    if (u_panel == 0)
    {
        c = texelFetch(u_albedo, t, 0).rgb;
    }
    else if (u_panel == 1)
    {
        c = texelFetch(u_data, t, 0).rgb;
    }
    else if (u_panel == 2)
    {
        uvec4 v = texelFetch(u_ids, t, 0);
        c = vec3(float(v.x) / 127.0, float(v.y) / 127.0, float(v.w & 255u) / 255.0);
    }
    else
    {
        ivec4 v = texelFetch(u_signed, t, 0);
        c = vec3(float(-v.x - 1) / 16129.0, float(v.y & 255) / 127.0, v.y < 0 ? 0.4 : 0.0);
    }
    o_color = vec4(c, 1.0);
}
