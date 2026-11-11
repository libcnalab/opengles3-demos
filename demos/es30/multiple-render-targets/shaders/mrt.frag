#version 300 es
// One fragment shader, four render targets. The location qualifier of each output is the index of
// the draw buffer it is written to (GLSL ES 3.00 4.3.8.2); glDrawBuffers then says which
// attachment of the framebuffer object each draw buffer is.
//
//   location 0  vec4   -> COLOR_ATTACHMENT0  RGBA8     (normalized fixed point: converted with eq. 2.3)
//   location 1  vec4   -> COLOR_ATTACHMENT1  RGBA8
//   location 2  uvec4  -> COLOR_ATTACHMENT2  RGBA32UI  (unsigned integers: stored unmodified)
//   location 3  ivec4  -> COLOR_ATTACHMENT3  RG32I     (signed integers; only x and y are stored)
//
// The value of every output is a pure function of the integer pixel coordinate, so the CPU can
// predict every pixel of every attachment exactly (integers) or to 8-bit precision (floats).
precision highp float;
precision highp int;

layout(location = 0) out vec4 o_albedo;
layout(location = 1) out vec4 o_data;
layout(location = 2) out uvec4 o_ids;
layout(location = 3) out ivec4 o_signed;

uniform ivec2 u_size;  // size of the render targets in pixels

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);             // pixel (x, y); gl_FragCoord is at the pixel centre
    vec2 uv = gl_FragCoord.xy / vec2(u_size);     // (x + 0.5) / width, exact: the width is a power of two

    // 0: a colour gradient on a 16x16 checkerboard (dark squares at half brightness).
    bool dark = (((p.x >> 4) + (p.y >> 4)) & 1) == 1;
    vec3 rgb = vec3(uv.x, uv.y, 0.25);
    if (dark) rgb *= 0.5;
    o_albedo = vec4(rgb, 1.0);

    // 1: a second normalized target: inverted gradient, mean of the coordinates, alpha 0.75.
    o_data = vec4(1.0 - uv.x, 1.0 - uv.y, 0.5 * (uv.x + uv.y), 0.75);

    // 2: unsigned integers that need all 32 bits (high bits set, a counter running downwards).
    o_ids = uvec4(uint(p.x), uint(p.y), 0xDEAD0000u | (uint(p.x) << 8u) | uint(p.y),
                  0xFFFFFFFFu - uint(p.y * u_size.x + p.x));

    // 3: signed integers: a negative product and a value close to INT_MIN. The attachment has only
    // two components, so z and w are dropped by the framebuffer.
    o_signed = ivec4(-(p.x * p.y) - 1, -2147483647 - 1 + p.x, 7, 7);
}
