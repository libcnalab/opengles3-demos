#version 300 es
// Exact readback path: the viewport is exactly the texture level (one fragment per texel), so the
// fragment centre (i + 0.5, j + 0.5) / size is the centre of texel (i, j) and NEAREST selects it.
// The sampled value is written bit for bit into an RGBA32UI target with floatBitsToUint.
precision highp float;

uniform highp sampler2D u_tex2d;
uniform highp sampler2DArray u_array;
uniform bool u_is_array;
uniform float u_layer;
uniform vec2 u_size;

layout(location = 0) out highp uvec4 o_bits;

void main()
{
    vec2 uv = gl_FragCoord.xy / u_size;
    vec4 t = u_is_array ? texture(u_array, vec3(uv, u_layer)) : texture(u_tex2d, uv);
    o_bits = floatBitsToUint(t);
}
