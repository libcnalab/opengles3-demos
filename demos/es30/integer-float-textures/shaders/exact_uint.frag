#version 300 es
// Reads an UNSIGNED-INTEGER texture through a usampler2D and writes the value it got into an
// RGBA32UI render target unchanged, so the application can read the exact 32-bit words back.
// The viewport is the size of the texture (8 x 8): every fragment is one texel.
//
// The sampler precision is spelled out: a sampler without a precision qualifier is lowp in the
// fragment language (GLSL ES 3.00 4.5.3), and a lowp lookup is not required to deliver more than
// about 9 bits. highp is what makes 32-bit integers come back whole.
precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D u_tex;
uniform int u_mode;        // 0: texture(), 1: texelFetch(), 2: textureLod()
uniform int u_lod;         // mip level for modes 1 and 2
uniform vec2 u_uv_shift;   // added to the coordinate of mode 0 (texture size 8: 0.0625 is half a texel)

layout(location = 0) out uvec4 o_value;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);                 // texel (i, j) of the 8 x 8 level 0
    vec2 uv = (vec2(p) + 0.5) / 8.0;                  // its center, exactly representable
    if (u_mode == 0)
        o_value = texture(u_tex, uv + u_uv_shift);
    else if (u_mode == 1)
        o_value = texelFetch(u_tex, p >> u_lod, u_lod);
    else
        o_value = textureLod(u_tex, uv, float(u_lod));
}
