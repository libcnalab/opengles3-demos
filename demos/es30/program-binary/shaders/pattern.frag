#version 300 es
// Truchet tiles: every square cell holds two quarter-circle arcs around opposite corners; a per-cell
// hash picks one of the two orientations, so the arcs join into long meandering curves.
// All arithmetic is integer (highp, wrapping uint multiplication per GLSL ES 3.00 §4.1.3), so the C++
// reference in main.cpp reproduces every pixel bit-exactly.

precision highp float;
precision highp int;

uniform ivec2 u_origin;  // window-space origin of the viewport this draw fills
uniform int   u_cell;    // cell size in pixels (> 0)
uniform uint  u_seed;    // hash seed: changes the tile orientations
uniform bool  u_gray;    // luminance-only output

out vec4 o_color;

const uvec3 kPalette[6] = uvec3[6](
    uvec3( 24,  38,  64),   // 0 night blue   (background)
    uvec3( 33,  52,  84),   // 1 slate blue   (background)
    uvec3( 42, 157, 143),   // 2 teal         (arcs)
    uvec3(233, 196, 106),   // 3 sand
    uvec3(244, 162,  97),   // 4 apricot
    uvec3(231, 111,  81));  // 5 coral

uint cell_hash(uvec2 c, uint seed)
{
    uint h = (c.x * 0x8DA6B343u) ^ (c.y * 0xD8163841u) ^ seed;
    h ^= h >> 13u;
    h *= 0x5BD1E995u;
    h ^= h >> 15u;
    return h;
}

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy) - u_origin;  // pixel index inside the viewport
    ivec2 c = p / u_cell;                          // cell index
    ivec2 l = p - c * u_cell;                      // pixel inside the cell
    uint h = cell_hash(uvec2(c), u_seed);
    if ((h & 1u) != 0u) l.x = u_cell - 1 - l.x;    // mirrored tile = the other orientation

    // Squared distance (in half-pixel units) from the pixel centre to the corners (0,0) and (s,s).
    int x2 = 2 * l.x + 1;
    int y2 = 2 * l.y + 1;
    int s2 = 2 * u_cell;
    int d0 = x2 * x2 + y2 * y2;
    int d1 = (s2 - x2) * (s2 - x2) + (s2 - y2) * (s2 - y2);
    int d = min(d0, d1);

    // Arc of radius s/2 (= u_cell half-pixels) and half-width w half-pixels.
    int w = max(1, u_cell / 4);
    bool on_arc = d >= (u_cell - w) * (u_cell - w) && d < (u_cell + w) * (u_cell + w);

    uvec3 bg = kPalette[int((h >> 8u) & 1u)];
    uvec3 fg = kPalette[2 + ((c.x + c.y) / 3) % 4];  // diagonal colour bands
    uvec3 col = on_arc ? fg : bg;
    if (u_gray)
    {
        uint g = (col.r * 77u + col.g * 150u + col.b * 29u) >> 8u;
        col = uvec3(g);
    }
    o_color = vec4(vec3(col) / 255.0, 1.0);
}
