#version 300 es
precision highp float;
precision highp int;
// Fragment shaders get no default precision for floats, and samplers default to lowp.
precision highp sampler2D;

uniform sampler2D u_tex;
uniform float u_t;    // constant t coordinate: all rows of a strip are identical
uniform int u_mode;   // 0: texture(), 1: encode textureSize(u_tex, u_lod) as a color
uniform int u_lod;

in float v_s;
out vec4 o_color;

void main()
{
    // u_mode is a uniform, so the branch is uniform control flow and the implicit derivatives
    // that texture() needs are well defined (GLSL ES 3.00 section 8.8).
    if (u_mode == 0)
    {
        o_color = texture(u_tex, vec2(v_s, u_t));
    }
    else
    {
        // textureSize returns the size of level (BASE_LEVEL + lod), ES 3.0 section 2.12.9.3. Sizes
        // up to 255 are exactly representable as an 8-bit color channel.
        ivec2 size = textureSize(u_tex, u_lod);
        o_color = vec4(float(size.x) / 255.0, float(size.y) / 255.0, 0.0, 1.0);
    }
}
