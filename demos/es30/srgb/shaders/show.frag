#version 300 es
// One band of the picture: a 256 x 1 row of a texture stretched over the window width, shown as gray. Window column x shows
// texel x * 256 / u_width (integer arithmetic, so the texel of every column is exact; at a width of 256 it is texel x).
//   u_channel 0 shows red, u_channel 3 shows alpha.
//   u_encode  applies the sRGB encoding formula (linear -> sRGB, equation 4.1 / its usual form with the
//             exponent 1/2.4) to the value before it is written, undoing a decode done by the sampler.
// Whether a lookup of an SRGB8_ALPHA8 texture returns the stored byte or its linearization is decided
// by the texture's internal format alone: this shader is the same for every band.
precision highp float;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform int u_row;
uniform int u_channel;
uniform bool u_encode;
uniform int u_width;    // window width in pixels

out vec4 o_color;

float linear_to_srgb(float c)
{
    return c < 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

void main()
{
    vec4 v = texelFetch(u_tex, ivec2(int(gl_FragCoord.x) * 256 / u_width, u_row), 0);
    float c = (u_channel == 3) ? v.a : v.r;
    if (u_encode) c = linear_to_srgb(c);
    o_color = vec4(vec3(c), 1.0);
}
