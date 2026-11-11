#version 320 es
// A pattern of floating-point values that every color-renderable float format of ES 3.2 can store exactly,
// and that lies largely OUTSIDE [0, 1]. Pixel (x, y) of a 16 x 16 target:
//   R = +-1.5 * 2^(x-4)           (sign: + for even y, - for odd y)  0.09375 ... 3072
//   G = (x - 8) * 0.5 + (y - 8) * 0.125                              -4.875 ... +4.375
//   B = 3.75 for even y, -2.5 for odd y                              (HDR white and a negative)
//   A = 0.25 * (x + 1)                                               0.25 ... 4.0
// The pixels (15,15) and (14,15) are replaced by the uniforms u_probe0 / u_probe1 (range extremes and a
// value that is not representable); the CPU model in main.cpp repeats all of this.
precision highp float;
precision highp int;

uniform vec4 u_probe0;
uniform vec4 u_probe1;

layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (p == ivec2(15, 15)) { o_color = u_probe0; return; }
    if (p == ivec2(14, 15)) { o_color = u_probe1; return; }

    float r = 0.09375;  // 1.5 * 2^-4
    for (int i = 0; i < 16; ++i)
    {
        if (i < p.x) r *= 2.0;  // exact: multiplication by a power of two
    }
    bool odd = (p.y & 1) == 1;
    float red = odd ? -r : r;
    float green = float(p.x - 8) * 0.5 + float(p.y - 8) * 0.125;
    float blue = odd ? -2.5 : 3.75;
    float alpha = 0.25 * float(p.x + 1);
    o_color = vec4(red, green, blue, alpha);
}
