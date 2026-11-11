#version 310 es
// A bar chart of the 16-bin histogram that 65536 invocations built with atomicAdd (words 16..31 of the Acc block).
// One 16-pixel column per bin; the bar is u_scale pixels per count high. Pixels above the bar are discarded, so
// the clear colour shows there.
precision highp float;
precision highp int;

layout(std430, binding = 0) readonly buffer Acc { uint v[]; } acc;

uniform float u_scale;    // pixels per count
uniform float u_y0;       // window y of the bottom of the viewport

out vec4 o_color;

void main()
{
    int x = int(gl_FragCoord.x);
    int bin = x / 16;
    if (x % 16 == 0) discard;                                   // one dark pixel between the bars
    float height = float(acc.v[16 + bin]) * u_scale;
    if (gl_FragCoord.y - u_y0 > height) discard;
    float t = float(bin) / 15.0;
    o_color = vec4(t, 0.6, 1.0 - t, 1.0);
}
