#version 320 es
// Draws one region of the window. The region is given in window pixels (u_origin, u_extent) and every source is
// addressed in normalized region coordinates, so the picture fills a window of any size.
//   u_mode 0  the advanced-blend grid (premultiplied RGBA8) over a checkerboard: C + (1 - A) * checker
//   u_mode 1  a multisample texture resolved by averaging its samples
//   u_mode 2  a multisample texture "exploded": every texel is split into sub-cells, sub-cell s shows sample s
//   u_mode 3  an empty panel (no multisample support)
precision highp float;
precision highp int;

uniform int u_mode;
uniform vec2 u_origin;
uniform vec2 u_extent;
uniform highp sampler2D u_grid;
uniform highp sampler2DMS u_ms;
uniform ivec2 u_texels;
uniform int u_samples;
uniform int u_cols;
uniform ivec2 u_cell;     // cell size of the grid in texels (cell borders are darkened)

out vec4 o_color;

void main()
{
    vec2 t = clamp((gl_FragCoord.xy - u_origin) / u_extent, 0.0, 0.99999);
    if (u_mode == 0)
    {
        ivec2 size = textureSize(u_grid, 0);
        ivec2 texel = ivec2(t * vec2(size));
        vec4 c = texelFetch(u_grid, texel, 0);
        ivec2 check = (texel / 8) % 2;
        vec3 board = (check.x ^ check.y) == 0 ? vec3(0.86) : vec3(0.70);
        vec3 rgb = c.rgb + (1.0 - c.a) * board;
        ivec2 inside = texel % u_cell;
        if (inside.x == 0 || inside.y == 0) rgb *= 0.35;
        o_color = vec4(rgb, 1.0);
    }
    else if (u_mode == 1)
    {
        ivec2 texel = ivec2(t * vec2(u_texels));
        vec4 sum = vec4(0.0);
        for (int s = 0; s < u_samples; ++s) sum += texelFetch(u_ms, texel, s);
        o_color = vec4(sum.rgb / float(u_samples), 1.0);
    }
    else if (u_mode == 2)
    {
        vec2 f = t * vec2(u_texels);
        ivec2 texel = ivec2(f);
        vec2 sub = fract(f);
        int rows = (u_samples + u_cols - 1) / u_cols;
        int s = int(sub.y * float(rows)) * u_cols + int(sub.x * float(u_cols));
        vec3 rgb = s < u_samples ? texelFetch(u_ms, texel, s).rgb : vec3(0.0);
        if (sub.x < 0.08 || sub.y < 0.08) rgb *= 0.3;
        o_color = vec4(rgb, 1.0);
    }
    else
    {
        o_color = vec4(0.18, 0.18, 0.22, 1.0);
    }
}
