#version 320 es
// Shows one layer of the 2D array texture on a tile of any size: v_ndc is the position inside the tile, (u, v) = v_ndc / 2 + 1 / 2
// in [0,1]^2 are the texture coordinates, and the layer index is the third coordinate of the lookup. The sampler object is NEAREST,
// so every tile pixel shows exactly one texel and the tile is the layer, scaled by whole pixels.
precision highp float;
precision highp sampler2DArray;

layout(binding = 0) uniform sampler2DArray u_array;
layout(location = 0) uniform int u_layer;

in vec2 v_ndc;
out vec4 frag_color;

void main()
{
    frag_color = texture(u_array, vec3(v_ndc * 0.5 + 0.5, float(u_layer)));
}
