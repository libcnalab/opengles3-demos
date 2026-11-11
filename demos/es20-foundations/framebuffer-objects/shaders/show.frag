#version 300 es
// Second pass of render-to-texture: sample the texture that was just rendered into.
//   u_lod < 0  : texture() with the sampler state of the texture (NEAREST here)
//   u_lod >= 0 : texelFetch() of one mipmap level (the level-2 attachment, which is only 16x16);
//                u_lod_size is that level's edge length in texels
//   u_invert   : 1.0 turns the image into its photographic negative (a trivial "post-process" pass
//                whose result is rendered into another texture)
//   u_gray     : 1.0 shows the first component (red) in all three channels; this is how a depth
//                texture, which returns its depth in .r, is made visible
// The branch is on a uniform, so every invocation takes the same path and the implicit derivatives
// that texture() needs stay well defined (GLSL ES 3.00 section 8.8).
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D u_tex;
uniform int u_lod;
uniform float u_lod_size;
uniform float u_invert;
uniform float u_gray;
in vec2 v_uv;
out vec4 o_color;
void main()
{
    vec4 c;
    if (u_lod >= 0)
        c = texelFetch(u_tex, ivec2(v_uv * u_lod_size), u_lod);
    else
        c = texture(u_tex, v_uv);
    vec3 rgb = (u_gray > 0.5) ? c.rrr : c.rgb;
    o_color = vec4(abs(vec3(u_invert) - rgb), 1.0);
}
