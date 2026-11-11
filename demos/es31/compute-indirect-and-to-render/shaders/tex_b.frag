#version 310 es
// Read half: a texelFetch of the texture that tex_a.frag wrote through an image, at the same pixel.
precision highp float;
uniform highp sampler2D u_tex;
layout(location = 0) out vec4 o_color;

void main()
{
    float t = texelFetch(u_tex, ivec2(gl_FragCoord.xy), 0).x;
    o_color = vec4(t, 1.0 - t, 0.5, 1.0);
}
