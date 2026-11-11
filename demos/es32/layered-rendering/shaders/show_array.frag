#version 320 es
// Shows one layer of the 2D array texture, texel for texel (texelFetch: no filtering, no interpolation).
precision highp float;
precision highp sampler2DArray;

layout(binding = 0) uniform sampler2DArray u_array;
layout(location = 0) uniform ivec2 u_origin;  // window position of the tile's lower left corner
layout(location = 1) uniform int u_layer;

out vec4 frag_color;

void main()
{
    frag_color = texelFetch(u_array, ivec3(ivec2(gl_FragCoord.xy) - u_origin, u_layer), 0);
}
