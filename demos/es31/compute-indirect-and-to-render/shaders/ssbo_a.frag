#version 310 es
// Write half of the "buffer store -> buffer load" pair in fragment shaders (needs MAX_FRAGMENT_SHADER_STORAGE_BLOCKS > 0;
// ES 3.1 allows 0). One word per window pixel; the window is 256 pixels wide.
precision highp float;
precision highp int;
layout(std430, binding = 0) buffer Data { uint v[]; } u_data;
uniform uint u_stride;  // words per row = window width
layout(location = 0) out vec4 o_color;

void main()
{
    uint x = uint(gl_FragCoord.x);
    uint y = uint(gl_FragCoord.y);
    u_data.v[y * u_stride + x] = x * 11u + y * 17u + 3u;
    o_color = vec4(0.0);
}
