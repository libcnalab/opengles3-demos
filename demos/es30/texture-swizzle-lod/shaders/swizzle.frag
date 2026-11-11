#version 300 es
// Returns the raw result of a texture lookup. The shader never swizzles: whatever differs between
// two cells that run this very program comes from the TEXTURE_SWIZZLE_* state of the bound texture
// (ES 3.0 3.8.7, 3.9.2.1), which is applied to every kind of lookup below.
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;   // 4 x 4 texels, NEAREST (or the sampler bound to the unit)
uniform int u_lookup;      // 0 texture, 1 texelFetch, 2 textureLod, 3 textureProj

in vec2 v_corner;
out vec4 o_color;

void main()
{
    // u_lookup is a uniform, so the branches below are dynamically uniform and the implicit
    // derivatives of texture() / textureProj() stay defined (GLSL ES 3.00 8.8).
    if (u_lookup == 0)
        o_color = texture(u_tex, v_corner);
    else if (u_lookup == 1)
        o_color = texelFetch(u_tex, ivec2(v_corner * 4.0), 0);   // 4 texels over the viewport
    else if (u_lookup == 2)
        o_color = textureLod(u_tex, v_corner, 0.0);
    else
        o_color = textureProj(u_tex, vec3(v_corner * 2.0, 2.0));  // (s, t, q) -> (s / q, t / q)
}
