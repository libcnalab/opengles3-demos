#version 300 es
// Same as exact_uint.frag for SIGNED-INTEGER textures: isampler2D in, RGBA32I render target out.
precision highp float;
precision highp int;
precision highp isampler2D;

uniform isampler2D u_tex;
uniform int u_mode;        // 0: texture(), 1: texelFetch(), 2: textureLod()
uniform int u_lod;
uniform vec2 u_uv_shift;

layout(location = 0) out ivec4 o_value;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 uv = (vec2(p) + 0.5) / 8.0;
    if (u_mode == 0)
        o_value = texture(u_tex, uv + u_uv_shift);
    else if (u_mode == 1)
        o_value = texelFetch(u_tex, p >> u_lod, u_lod);
    else
        o_value = textureLod(u_tex, uv, float(u_lod));
}
