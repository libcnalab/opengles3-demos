#version 300 es
// Samples a texture that was filled by glCopyTexImage2D / glCopyTexSubImage2D, with NEAREST filtering,
// so that every texel can be compared with the framebuffer pixel it was copied from.
//   u_mode 0 : (r, g, b), alpha forced to 1   - what a LUMINANCE texture looks like: (L, L, L)
//   u_mode 1 : (a, a, a), alpha forced to 1   - makes an ALPHA texture visible
//   u_mode 2 : (r, g, b, a) untouched         - used for the numeric checks
// Texture formats without all four components are expanded by the GL: missing G and B are 0 and a
// missing A is 1, except LUMINANCE which returns (L, L, L, 1) and ALPHA which returns (0, 0, 0, A)
// (ES 3.0 Table 3.24).
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D u_tex;
uniform int u_mode;
in vec2 v_uv;
out vec4 o_color;
void main()
{
    vec4 c = texture(u_tex, v_uv);
    if (u_mode == 1)
        o_color = vec4(c.aaa, 1.0);
    else if (u_mode == 0)
        o_color = vec4(c.rgb, 1.0);
    else
        o_color = c;
}
