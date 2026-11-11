#version 320 es
precision highp float;
precision highp int;

// Fragment-shader texelFetch: the strip shows the per-instance colour texels (upper half) and the per-instance
// material texels (lower half) of the buffer textures, one cell per instance.
uniform highp samplerBuffer u_color;
uniform highp usamplerBuffer u_material;
uniform ivec4 u_rect;  // x, y, width, height of the strip in window coordinates
uniform int u_cells;

out vec4 o_color;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy) - u_rect.xy;
    int cell = clamp(p.x * u_cells / u_rect.z, 0, u_cells - 1);
    if (p.y >= u_rect.w / 2)
    {
        o_color = vec4(texelFetch(u_color, cell).rgb, 1.0);
    }
    else
    {
        float g = float(texelFetch(u_material, cell).x + 1u) * 0.25;  // material 0, 1, 2 -> 0.25, 0.5, 0.75
        o_color = vec4(g, g, g, 1.0);
    }
}
