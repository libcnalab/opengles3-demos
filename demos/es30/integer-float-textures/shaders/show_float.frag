#version 300 es
// Draws one cell for a floating-point texture (R32F, RGBA32F, RGBA16F, R16F, R11F_G11F_B10F, RGB9_E5):
//     color = clamp(value * u_scale + u_bias, 0, 1)
// u_use_texture selects texelFetch() (0) or a filtered-capable texture() lookup at the texel center (1).
// The latter is used with a sampler object whose filters are LINEAR, which makes an R32F texture
// complete or incomplete depending on the implementation (see the README).
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D u_tex;
uniform ivec2 u_origin;
uniform ivec2 u_cell_size;  // pixel p of a cell n pixels wide shows texel p * 8 / n (integer division)
uniform vec3 u_scale;
uniform vec3 u_bias;
uniform bool u_gray;
uniform int u_use_texture;

out vec4 o_color;

void main()
{
    ivec2 t = (ivec2(gl_FragCoord.xy) - u_origin) * 8 / u_cell_size;
    vec4 v = (u_use_texture == 1) ? texture(u_tex, (vec2(t) + 0.5) / 8.0) : texelFetch(u_tex, t, 0);
    vec3 c = u_gray ? vec3(v.r) : v.rgb;
    o_color = vec4(clamp(c * u_scale + u_bias, 0.0, 1.0), 1.0);
}
