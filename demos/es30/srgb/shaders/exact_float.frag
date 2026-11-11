#version 300 es
// Samples a texture and writes the four 32-bit floats it got, bit for bit, into an RGBA32UI render
// target (floatBitsToUint). The viewport is 256 x 1: fragment x is the texel index of a 256 x 1 texture.
// Reading the words back with glReadPixels(RGBA_INTEGER, UNSIGNED_INT) shows the decoded values
// without the rounding of an 8-bit render target. RGBA32UI is color-renderable in core ES 3.0
// (Table 3.13); the float render targets are not.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform int u_mode;     // 0: texture() at the texel center, 1: texelFetch(), 2: texture() at u_uv
uniform vec2 u_uv;      // mode 2: one fixed coordinate for every fragment

layout(location = 0) out uvec4 o_value;

void main()
{
    int x = int(gl_FragCoord.x);
    vec4 v;
    if (u_mode == 0)
        v = texture(u_tex, vec2((float(x) + 0.5) / 256.0, 0.5));
    else if (u_mode == 1)
        v = texelFetch(u_tex, ivec2(x, 0), 0);
    else
        v = texture(u_tex, u_uv);
    o_value = floatBitsToUint(v);
}
