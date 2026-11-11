#version 300 es
// Read-back path for the shadow map. glReadPixels cannot return depth in OpenGL ES (ES 3.0 4.3.2),
// so selected texels are fetched with texelFetch and written as 24-bit integers into the RGB
// channels of an RGBA8 render target, one output pixel per requested texel.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_depth;   // comparison mode NONE through a sampler object
uniform ivec2 u_texels[16];  // texel coordinates to fetch, indexed by the output pixel x

out vec4 o_color;

void main()
{
    ivec2 texel = u_texels[int(gl_FragCoord.x)];
    float d = texelFetch(u_depth, texel, 0).r;
    uint v = uint(d * 16777215.0 + 0.5);
    o_color = vec4(float((v >> 16) & 255u), float((v >> 8) & 255u), float(v & 255u), 255.0) / 255.0;
}
