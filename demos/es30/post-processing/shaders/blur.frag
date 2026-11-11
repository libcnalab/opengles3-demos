#version 300 es
// One direction of a separable Gaussian blur. A 2D Gaussian is the product of two 1D Gaussians, so blurring
// along x and then along y gives the same result as one (2r+1) x (2r+1) kernel at 2(2r+1) instead of
// (2r+1)^2 texture fetches per pixel.
//
//   u_dir     (1,0) for the horizontal pass, (0,1) for the vertical pass
//   u_radius  taps at offsets -u_radius .. +u_radius
//   u_sigma   standard deviation in texels; the weights exp(-i^2 / (2 sigma^2)) are normalized by their sum
//
// texelFetch has no wrap mode: coordinates outside the texture would be undefined, so they are clamped by
// hand, which is the same as CLAMP_TO_EDGE (the edge texel is repeated).
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_src;
uniform ivec2 u_dir;
uniform int u_radius;
uniform float u_sigma;

layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 size = textureSize(u_src, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 sum = vec3(0.0);
    float weights = 0.0;
    for (int i = -u_radius; i <= u_radius; ++i)
    {
        float w = exp(-0.5 * float(i * i) / (u_sigma * u_sigma));
        ivec2 q = clamp(p + i * u_dir, ivec2(0), size - 1);
        sum += w * texelFetch(u_src, q, 0).rgb;
        weights += w;
    }
    o_color = vec4(sum / weights, 1.0);
}
