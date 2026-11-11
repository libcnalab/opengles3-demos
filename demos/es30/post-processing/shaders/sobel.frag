#version 300 es
// Sobel edge detection on the perceptual brightness of the scene. The 3x3 kernels
//
//        -1  0  +1            -1  -2  -1
//   Gx = -2  0  +2       Gy =  0   0   0
//        -1  0  +1            +1  +2  +1
//
// are the derivative [-1 0 1] along one axis smoothed by [1 2 1] along the other. The output is the gradient
// magnitude |(Gx, Gy)| mapped through smoothstep(u_low, u_high, .) to [0, 1], in all three channels: gradients
// below u_low (the soft shading of a sphere, the one-step bands of an 8-bit gradient) vanish, gradients above
// u_high (an object outline, a checker edge) are full white. The brightness is the Rec. 709 luma of the linear
// color, square-rooted as a cheap stand-in for the sRGB curve, so that edges in dark areas are not suppressed
// by the linear encoding. Neighbors outside the texture repeat the edge texel.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_src;
uniform float u_low;
uniform float u_high;

layout(location = 0) out vec4 o_color;

float brightness(ivec2 q, ivec2 size)
{
    vec3 c = texelFetch(u_src, clamp(q, ivec2(0), size - 1), 0).rgb;
    return sqrt(dot(c, vec3(0.2126, 0.7152, 0.0722)));
}

void main()
{
    ivec2 size = textureSize(u_src, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    float tl = brightness(p + ivec2(-1, 1), size), t = brightness(p + ivec2(0, 1), size), tr = brightness(p + ivec2(1, 1), size);
    float l = brightness(p + ivec2(-1, 0), size), r = brightness(p + ivec2(1, 0), size);
    float bl = brightness(p + ivec2(-1, -1), size), b = brightness(p + ivec2(0, -1), size), br = brightness(p + ivec2(1, -1), size);
    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (tl + 2.0 * t + tr) - (bl + 2.0 * b + br);
    float edge = smoothstep(u_low, u_high, sqrt(gx * gx + gy * gy));
    o_color = vec4(vec3(edge), 1.0);
}
