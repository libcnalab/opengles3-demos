#version 310 es
// The stencil index of the same textures (DEPTH_STENCIL_TEXTURE_MODE = STENCIL_INDEX) through an UNSIGNED
// integer sampler (ES 3.2 section 8.16: only NEAREST filtering is supported). The 8-bit value s is shown as
// the color (s, 37 s mod 256, 91 s mod 256) / 255, which the application reproduces on the CPU.
precision highp float;
precision highp int;

layout(binding = 0) uniform highp usampler2D u_stencil;

in vec2 v_uv;
out vec4 o_color;

void main()
{
    uint s = texture(u_stencil, v_uv).r;
    o_color = vec4(float(s) / 255.0, float((s * 37u) & 255u) / 255.0, float((s * 91u) & 255u) / 255.0, 1.0);
}
