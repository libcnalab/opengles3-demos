#version 310 es
// Writes pattern number u_sample at every pixel it covers: a color that depends on the pixel position and on the number,
// so that two different numbers never give the same color anywhere. The demo draws pattern k with the sample mask set to
// the single sample k: that gives every sample of a multisample texture its own, known content, whatever the (implementation
// dependent) positions of the samples are. The same function is written in C++ in main.cpp (pattern()).
//
// main.cpp injects #define lines after the #version line:
//   GVEC4       vec4 / ivec4 / uvec4: the type of the color output (it must match the attachment)
//   KIND_FLOAT | KIND_INT | KIND_UINT
precision highp float;
precision highp int;

uniform int u_sample;
out GVEC4 o_color;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    int k = u_sample;
    ivec4 v = ivec4(16 * p.x + 3 * k, 20 * p.y + 5 * k, 97 * k + 13 * (p.x + p.y), 200 + 7 * k + 3 * p.y) & 255;
#if defined(KIND_FLOAT)
    o_color = vec4(v) / 255.0;          // RGBA8: the byte v comes back exactly
#elif defined(KIND_INT)
    o_color = v - ivec4(128);           // RGBA8I: -128 .. 127
#else
    o_color = uvec4(v);                 // RGBA8UI: 0 .. 255
#endif
}
