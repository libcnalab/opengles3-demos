#version 300 es
// Shows one decoded ETC2/EAC texture region, magnified with NEAREST filtering.
// Exactly one of the two samplers is used per draw (u_is_array is uniform): unit 0 holds the
// TEXTURE_2D textures, unit 1 the TEXTURE_2D_ARRAY texture.
precision highp float;

uniform highp sampler2D u_tex2d;
uniform highp sampler2DArray u_array;
uniform bool u_is_array;
uniform float u_layer;
// 0: RGB as is. 1: RGBA composited over a checkerboard (formats with alpha).
// 2: unsigned R11/RG11 shown as (r, g, 1 - r). 3: signed R11/RG11, remapped to [0, 1] first.
uniform int u_mode;

in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
    vec4 t = u_is_array ? texture(u_array, vec3(v_uv, u_layer)) : texture(u_tex2d, v_uv);
    if (u_mode == 0)
    {
        o_color = vec4(t.rgb, 1.0);
    }
    else if (u_mode == 1)
    {
        ivec2 p = ivec2(gl_FragCoord.xy) / 6;
        float checker = ((p.x + p.y) % 2 == 0) ? 77.0 / 255.0 : 158.0 / 255.0;
        o_color = vec4(mix(vec3(checker), t.rgb, t.a), 1.0);
    }
    else
    {
        vec2 s = (u_mode == 3) ? 0.5 + 0.5 * t.rg : t.rg;
        o_color = vec4(s.x, s.y, 1.0 - s.x, 1.0);
    }
}
