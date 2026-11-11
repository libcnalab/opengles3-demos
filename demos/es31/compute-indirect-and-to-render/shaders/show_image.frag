#version 310 es
// Shows an RGBA8 texture (written by compute shaders) with texelFetch, u_scale pixels per texel, in the cell whose
// lower-left window pixel is u_origin.
precision highp float;
uniform highp sampler2D u_tex;
uniform ivec2 u_origin;
uniform int u_scale;
layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 t = (ivec2(gl_FragCoord.xy) - u_origin) / u_scale;
    o_color = vec4(texelFetch(u_tex, t, 0).rgb, 1.0);
}
