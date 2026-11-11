#version 310 es
// textureSize on a multisample sampler takes no level of detail: textureSize(gsampler2DMS) (GLSL ES 3.10 section 8.9.1).
// The size goes to a signed-integer target (one pixel) and is read back.
//
// main.cpp injects:  SAMPLER  e.g. "highp usampler2DMS"
precision highp float;
precision highp int;

uniform SAMPLER u_ms;
out ivec4 o_size;

void main()
{
    o_size = ivec4(textureSize(u_ms), 0, 0);
}
