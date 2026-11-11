#version 310 es
// Shows every sample of a multisample texture in one 128 x 128 window cell. The cell is a grid of u_cols x u_cols sub-cells,
// sub-cell s (row by row from the lower left) shows sample s with its texels stretched to fill the sub-cell.
//
// main.cpp injects #define lines after the #version line:
//   SAMPLER   e.g. "highp sampler2DMS"
//   GVEC4     vec4 / ivec4 / uvec4
//   KIND_FLOAT | KIND_INT | KIND_UINT   how a fetched texel becomes a color (the integer patterns hold 0..255, or -128..127)
precision highp float;
precision highp int;

uniform SAMPLER u_ms;
uniform ivec2 u_texels;   // width and height of the multisample texture
uniform ivec2 u_origin;   // lower left pixel of the cell in the window
uniform int u_cell;       // edge of the cell in pixels
uniform int u_samples;    // samples per texel
uniform int u_cols;       // sub-cells per row and per column

out vec4 o_color;

void main()
{
    ivec2 local = ivec2(gl_FragCoord.xy) - u_origin;
    int sub = u_cell / u_cols;
    ivec2 cell = local / sub;
    int k = cell.y * u_cols + cell.x;
    if (cell.x >= u_cols || cell.y >= u_cols || k >= u_samples)
    {
        o_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    ivec2 texel = min((local - cell * sub) * u_texels / sub, u_texels - 1);
    GVEC4 v = texelFetch(u_ms, texel, k);
#if defined(KIND_FLOAT)
    o_color = vec4(v.rgb, 1.0);
#elif defined(KIND_INT)
    o_color = vec4(vec3(v.rgb + ivec3(128)) / 255.0, 1.0);
#else
    o_color = vec4(vec3(v.rgb) / 255.0, 1.0);
#endif
}
