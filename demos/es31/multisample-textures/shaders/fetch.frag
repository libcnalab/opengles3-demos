#version 310 es
// Reads every sample of a multisample texture with texelFetch(sampler2DMS, texel, sample) (GLSL ES 3.10 section 8.9.2).
//
// The output image is a row of strips: the pixel (x, y) shows sample x / W of the texel (x % W, y), where W is the width of the
// multisample texture. Sample s of the whole texture is therefore the strip [s * W, (s + 1) * W). Rendered into a single-sample
// target the samples can be read back with glReadPixels, which cannot read a multisample framebuffer.
//
// main.cpp injects #define lines after the #version line:
//   SAMPLER   the declaration, e.g. "highp isampler2DMS" (GLSL ES 3.10 section 4.7.4 gives sampler2DMS no default precision)
//   GVEC4     vec4 / ivec4 / uvec4, the type of the fetched texel and of the output
precision highp float;
precision highp int;

uniform SAMPLER u_ms;
uniform ivec2 u_texels;   // width and height of the multisample texture

out GVEC4 o_value;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    o_value = texelFetch(u_ms, ivec2(p.x % u_texels.x, p.y), p.x / u_texels.x);
}
