#version 300 es
// Draws one cell for a SIGNED-INTEGER texture. u_mode 0: color = clamp(value * u_scale + u_bias, 0, 1),
// as in show_uint.frag. u_mode 1 is the "above 2^24" cell for an R32I texture that holds 16777216 + k
// (k = 0..63, row by row): every texel block is split in two halves,
//   left:  (value - 16777216) / 63 as gray - the integer is subtracted EXACTLY, so this is a clean ramp;
//   right: green when int(float(value)) == value, red otherwise - the same value taken through a
//          32-bit float. A float has 24 significant bits, so above 2^24 only every second integer
//          exists: odd values cannot survive the trip, whatever the rounding mode is.
precision highp float;
precision highp int;
precision highp isampler2D;

uniform isampler2D u_tex;
uniform ivec2 u_origin;
uniform vec3 u_scale;
uniform vec3 u_bias;
uniform bool u_gray;
uniform int u_mode;
uniform float u_zero;       // always 0.0; keeps the compiler from folding int(float(x)) back to x

out vec4 o_color;

void main()
{
    ivec2 local = ivec2(gl_FragCoord.xy) - u_origin;
    ivec2 t = local / 8;
    ivec4 v = texelFetch(u_tex, t, 0);
    if (u_mode == 1)
    {
        if ((local.x & 7) < 4)
            o_color = vec4(vec3(float(v.r - 16777216) / 63.0), 1.0);
        else
        {
            // u_zero is a uniform the compiler cannot see through: without it, Mesa's optimizer rewrites
            // int(float(x)) as x, which would hide exactly the property being demonstrated.
            float through_float = float(v.r) + u_zero;
            o_color = (int(through_float) == v.r) ? vec4(0.0, 1.0, 0.0, 1.0) : vec4(1.0, 0.0, 0.0, 1.0);
        }
        return;
    }
    vec3 c = u_gray ? vec3(float(v.r)) : vec3(v.rgb);
    o_color = vec4(clamp(c * u_scale + u_bias, 0.0, 1.0), 1.0);
}
