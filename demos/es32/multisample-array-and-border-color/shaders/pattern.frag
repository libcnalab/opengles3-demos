#version 320 es
// Writes a value that depends on the pixel, the layer and the sample into every sample of a
// multisample array layer. main.cpp inserts the defines after the #version line:
//   KIND_FLOAT | KIND_INT | KIND_UINT   the type of the color attachment (RGBA8, RGBA8I, RGBA8UI)
//   FROM_GS                             the layer comes from the geometry shader (layered rendering),
//                                       otherwise from u_layer (one glFramebufferTextureLayer per layer)
// pattern() is the same arithmetic as pattern() in main.cpp.
precision highp float;
precision highp int;

#ifdef FROM_GS
flat in highp int v_layer;
#else
uniform highp int u_layer;
#endif

#if defined(KIND_FLOAT)
layout(location = 0) out vec4 o_color;
#elif defined(KIND_INT)
layout(location = 0) out ivec4 o_color;
#else
layout(location = 0) out uvec4 o_color;
#endif

uvec4 pattern(uint x, uint y, uint layer, uint s)
{
    return uvec4((29u * x + 61u * layer + 17u * s + 11u) & 255u,
                 (37u * y + 53u * s + 23u * layer + 5u) & 255u,
                 (67u * layer + 41u * s + 5u * x + 3u * y) & 255u,
                 (250u - 23u * s - 13u * layer) & 255u);  // unsigned arithmetic wraps modulo 2^32 (GLSL ES 3.20 section 4.1.3)
}

void main()
{
#ifdef FROM_GS
    int layer = v_layer;
#else
    int layer = u_layer;
#endif
    // Reading gl_SampleID makes the whole shader run once per sample (ES 3.2 section 14.2.2), and each
    // invocation's output goes to its own sample: sample k of the texel receives pattern(..., k).
    // gl_FragCoord may hold the sample position, which is inside the pixel, so floor() gives the pixel.
    ivec2 pixel = ivec2(floor(gl_FragCoord.xy));
    uvec4 p = pattern(uint(pixel.x), uint(pixel.y), uint(layer), uint(gl_SampleID));
#if defined(KIND_FLOAT)
    o_color = vec4(p) / 255.0;  // k / 255 converts back to exactly k in an 8-bit normalized attachment
#elif defined(KIND_INT)
    o_color = ivec4(p) - 128;   // -128 .. 127 fits RGBA8I
#else
    o_color = p;
#endif
}
