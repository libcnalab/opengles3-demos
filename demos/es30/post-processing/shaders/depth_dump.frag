#version 300 es
// Makes the depth buffer readable. ES 3.0 has no glReadPixels of depth, so the 24-bit window-space depth of the
// scene is converted to the integer k = round(depth * (2^24 - 1)) and written as three bytes (red = bits
// 23..16, green = 15..8, blue = 7..0). The CPU rebuilds depth = k / (2^24 - 1). The sampler is highp: see the
// note in bright.frag.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_depth;

layout(location = 0) out vec4 o_color;

void main()
{
    float d = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;
    uint k = min(uint(d * 16777215.0 + 0.5), 16777215u);  // d = 1.0 would round to 2^24 in float
    o_color = vec4(float((k >> 16) & 255u), float((k >> 8) & 255u), float(k & 255u), 255.0) / 255.0;
}
