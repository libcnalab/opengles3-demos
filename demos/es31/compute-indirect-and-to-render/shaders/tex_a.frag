#version 310 es
// Write half of the "image store -> texture fetch" pair: one imageStore per pixel (r32f image, ES 3.2 Table 8.26).
precision highp float;
precision highp int;
layout(r32f, binding = 0) uniform writeonly highp image2D u_img;
layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    imageStore(u_img, p, vec4(float((uint(p.x) ^ uint(p.y)) & 127u) / 127.0));
    o_color = vec4(0.0);
}
