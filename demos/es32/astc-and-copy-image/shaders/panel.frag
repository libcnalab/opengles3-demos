#version 320 es
// Shows a compressed (ASTC) texture magnified with texelFetch: the panel's rectangle in window
// coordinates (u_origin, u_panel) is mapped onto the texture's u_texels texels, so every texel
// becomes a flat rectangle (nearest neighbour, no filtering, no LOD selection). Only RGB is shown;
// alpha is checked by the read-back passes.
precision highp float;
precision highp int;

uniform highp sampler2D u_tex;
uniform highp vec2 u_origin;   // lower-left corner of the panel (pixels)
uniform highp vec2 u_panel;    // panel size (pixels)
uniform highp ivec2 u_texels;  // texture size (texels)

out vec4 o_color;

void main()
{
    vec2 t = (gl_FragCoord.xy - u_origin) * vec2(u_texels) / u_panel;
    ivec2 texel = clamp(ivec2(floor(t)), ivec2(0), u_texels - 1);
    o_color = vec4(texelFetch(u_tex, texel, 0).rgb, 1.0);
}
