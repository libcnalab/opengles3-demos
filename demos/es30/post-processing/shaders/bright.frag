#version 300 es
// Bloom, step 1: the bright-pass. Keeps only what is brighter than u_threshold (per channel) and rescales it
// to [0, 1], so a dim surface contributes nothing to the glow and a white lamp contributes 1.
//
// All post-processing shaders read their input with texelFetch at ivec2(gl_FragCoord.xy): one texel per
// pixel, no filtering, no texture-coordinate arithmetic. The precision statements matter: a fragment
// shader's default sampler precision is lowp (GLSL ES 3.00 section 4.5.4), and the lookup result takes
// the precision of the sampler, so without `highp sampler2D` a 24-bit depth texture would come back with
// only the precision of lowp.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_src;
uniform float u_threshold;

layout(location = 0) out vec4 o_color;

void main()
{
    vec3 c = texelFetch(u_src, ivec2(gl_FragCoord.xy), 0).rgb;
    o_color = vec4(max(c - u_threshold, 0.0) / (1.0 - u_threshold), 1.0);
}
