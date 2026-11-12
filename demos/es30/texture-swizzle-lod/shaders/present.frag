#version 300 es
// Shows the contact sheet (an RGBA8 texture the cells were drawn into) stretched over the whole window.
// Window pixel (x, y) shows sheet texel (x * sheet width / window width, y * sheet height / window height),
// computed in integer arithmetic and fetched with texelFetch, so every window pixel is one exact sheet texel:
// no filtering, no rounding. Alpha is written as 1 so that the window is opaque (the sheet's own alpha is
// part of the cell checks, which read the sheet itself).
precision highp float;
precision highp int;

uniform highp sampler2D u_sheet;
uniform ivec2 u_window;   // window size in pixels

out vec4 o_color;

void main()
{
    ivec2 size = textureSize(u_sheet, 0);
    ivec2 texel = ivec2(gl_FragCoord.xy) * size / u_window;
    o_color = vec4(texelFetch(u_sheet, texel, 0).rgb, 1.0);
}
