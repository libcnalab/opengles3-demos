#version 310 es
// Read half: the fragment shader loads the word that ssbo_a.frag stored for the same pixel.
precision highp float;
precision highp int;
layout(std430, binding = 0) readonly buffer Data { uint v[]; } u_data;
uniform uint u_stride;  // words per row = window width
layout(location = 0) out vec4 o_color;

void main()
{
    uint x = uint(gl_FragCoord.x);
    uint y = uint(gl_FragCoord.y);
    uint v = u_data.v[y * u_stride + x];
    o_color = vec4(float(v % 256u) / 255.0, float(v / 256u) / 28.0, 0.5, 1.0);
}
