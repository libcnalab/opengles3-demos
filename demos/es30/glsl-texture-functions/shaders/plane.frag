#version 300 es
// Mip-level visualisation: texture() with an implicit level of detail on the RGBA8 test texture, whose texels encode
// (level, i, j) as R = 32 level + 16, G = 8 i, B = 8 j. The level is shown as a palette colour and the texel parity as
// a checkerboard, so the bands where the implementation switches levels are visible.
precision highp float;

uniform highp sampler2D u_t2d;
in vec2 v_uv;
out vec4 o_color;

const vec3 kPalette[8] = vec3[8](vec3(0.902, 0.235, 0.204), vec3(0.957, 0.612, 0.180), vec3(0.941, 0.886, 0.290),
                                 vec3(0.384, 0.769, 0.329), vec3(0.231, 0.702, 0.769), vec3(0.290, 0.416, 0.875),
                                 vec3(0.588, 0.341, 0.839), vec3(0.878, 0.878, 0.878));

void main()
{
    vec4 t = texture(u_t2d, v_uv);
    int level = clamp(int(round((t.r * 255.0 - 16.0) / 32.0)), 0, 7);
    int parity = (int(round(t.g * 255.0 / 8.0)) + int(round(t.b * 255.0 / 8.0))) & 1;
    o_color = vec4(kPalette[level] * (parity == 0 ? 1.0 : 0.85), 1.0);
}
