#version 310 es
// Reads sample u_sample of the RGBA8UI multisample texture at this pixel with texelFetch(usampler2DMS, texel, sample)
// (GLSL ES 3.10 section 8.9.2). Rendered into a single-sample RGBA8UI target of the same size, one pass per sample, so
// the CPU sees every stored sample without any resolve. ES 3.1+ contexts only.
precision highp float;
precision highp int;
uniform highp usampler2DMS u_ms;  // multisample samplers have no default precision (GLSL ES 3.10 section 4.7.4)
uniform int u_sample;
out uvec4 o_value;

void main()
{
    o_value = texelFetch(u_ms, ivec2(gl_FragCoord.xy), u_sample);
}
