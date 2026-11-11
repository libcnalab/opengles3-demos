#version 300 es
// Same for floating-point textures (R32F, RGBA16F, RGB9_E5, ...): sampler2D in. The result is a vec4
// of 32-bit floats; floatBitsToUint() copies its bit patterns into an RGBA32UI render target, which
// is the only way in core ES 3.0 to look at a float value without any further rounding (the float
// render targets need an extension, and an 8-bit target quantizes).
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform int u_mode;        // 0: texture(), 1: texelFetch(), 2: textureLod()
uniform int u_lod;
uniform vec2 u_uv_shift;

layout(location = 0) out uvec4 o_value;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 uv = (vec2(p) + 0.5) / 8.0;
    vec4 v;
    if (u_mode == 0)
        v = texture(u_tex, uv + u_uv_shift);
    else if (u_mode == 1)
        v = texelFetch(u_tex, p >> u_lod, u_lod);
    else
        v = textureLod(u_tex, uv, float(u_lod));
    o_value = floatBitsToUint(v);
}
