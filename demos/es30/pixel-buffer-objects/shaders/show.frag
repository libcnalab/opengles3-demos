#version 300 es
// Shows a texture over the whole viewport. The viewport is `scale` times the texture size and the
// texture is sampled with NEAREST filtering, so pixel (x, y) shows texel (x / scale, y / scale) and every
// texel uploaded from a pixel buffer object can be compared with the pixel it produced.
precision highp float;
precision highp sampler2D;

uniform sampler2D u_tex;

in vec2 v_uv;
out vec4 o_color;

void main()
{
    o_color = texture(u_tex, v_uv);
}
