#version 320 es
// Shows a 2D multisample array texture as a grid of tiles: column = sample, row = layer (layer 0 at
// the top). Each tile is the texture's texels scaled up, read with texelFetch, so what is on screen is
// one sample of one layer, unresolved. A 2-pixel gap on the right and at the bottom of every tile
// keeps the clear color. main.cpp computes the same integer arithmetic for its pixel reference.
precision highp float;
precision highp int;

uniform highp sampler2DMSArray u_ms;
uniform highp ivec2 u_origin;   // window position of the grid's bottom-left corner
uniform highp ivec2 u_tile;     // tile size in pixels, gap included
uniform highp ivec2 u_texels;   // texture width and height
uniform highp int u_layers;

layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 p = ivec2(floor(gl_FragCoord.xy)) - u_origin;
    ivec2 tile = p / u_tile;
    ivec2 local = p - tile * u_tile;
    if (local.x >= u_tile.x - 2 || local.y < 2)
        discard;
    int s = tile.x;
    int layer = u_layers - 1 - tile.y;
    ivec2 texel = ivec2(local.x * u_texels.x / (u_tile.x - 2), (local.y - 2) * u_texels.y / (u_tile.y - 2));
    o_color = vec4(texelFetch(u_ms, ivec3(texel, layer), s).rgb, 1.0);
}
