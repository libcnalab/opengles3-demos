#version 320 es
// Four fragment outputs, one per draw buffer, each a uniform color: every draw buffer receives its own source
// color, so that the per-draw-buffer blend state (glBlendFunci, glColorMaski, ...) decides four different
// results from one draw. The values are plain uniforms (no arithmetic): the CPU knows them exactly.
precision highp float;

uniform vec4 u_src[4];

layout(location = 0) out vec4 o_color0;
layout(location = 1) out vec4 o_color1;
layout(location = 2) out vec4 o_color2;
layout(location = 3) out vec4 o_color3;

void main()
{
    o_color0 = u_src[0];
    o_color1 = u_src[1];
    o_color2 = u_src[2];
    o_color3 = u_src[3];
}
