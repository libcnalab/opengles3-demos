#version 320 es
// Shows a 2D multisample array texture as a grid of tiles: column = sample, row = layer (layer 0 at
// the top). Each tile is the texture's texels scaled up, read with texelFetch, so what is on screen is
// one sample of one layer, unresolved. A gap of u_gap pixels on the right and at the bottom of every
// tile keeps the clear color. The tile size comes from the surface size (main.cpp, Layout); everything
// here is integer arithmetic on the pixel position, and main.cpp repeats it for its pixel reference.
precision highp float;
precision highp int;

uniform highp sampler2DMSArray u_ms;
uniform highp ivec2 u_origin;   // window position of the grid's bottom-left corner
uniform highp ivec2 u_tile;     // tile size in pixels, gap included
uniform highp int u_gap;        // gap in pixels
uniform highp int u_columns;    // number of samples shown
uniform highp ivec2 u_texels;   // texture width and height
uniform highp int u_layers;

layout(location = 0) out vec4 o_color;

void main()
{
    ivec2 p = ivec2(floor(gl_FragCoord.xy)) - u_origin;
    ivec2 tile = p / u_tile;
    ivec2 local = p - tile * u_tile;
    // Pixels right of the last column or above the last row (the remainder of the division) and the gaps keep the clear color.
    if (tile.x >= u_columns || tile.y >= u_layers || local.x >= u_tile.x - u_gap || local.y < u_gap)
        discard;
    int s = tile.x;
    int layer = u_layers - 1 - tile.y;
    ivec2 texel = ivec2(local.x * u_texels.x / (u_tile.x - u_gap), (local.y - u_gap) * u_texels.y / (u_tile.y - u_gap));
    o_color = vec4(texelFetch(u_ms, ivec3(texel, layer), s).rgb, 1.0);
}
