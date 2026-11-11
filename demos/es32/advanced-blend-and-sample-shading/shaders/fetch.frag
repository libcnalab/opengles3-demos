#version 320 es
// Second pass: copies every sample of a multisample texture into a single-sample strip with
// texelFetch(sampler2DMS, texel, sample) (GLSL ES 3.20 section 8.9.2), so that glReadPixels can read them.
// Pixel (x, y) of the strip holds sample x / W of texel (x % W, y).
precision highp float;
precision highp int;

uniform highp sampler2DMS u_ms;
uniform ivec2 u_texels;   // width and height of the multisample texture

out vec4 o_value;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    o_value = texelFetch(u_ms, ivec2(p.x % u_texels.x, p.y), p.x / u_texels.x);
}
