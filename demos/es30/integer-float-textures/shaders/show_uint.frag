#version 300 es
// Draws one 64 x 64 cell of the contact sheet: the 8 x 8 texels of an UNSIGNED-INTEGER texture,
// each as an 8 x 8 pixel block. The integer value is converted to float (this is only for display;
// the exact values are validated through exact_uint.frag) and mapped to a color with
//     color = clamp(value * u_scale + u_bias, 0, 1)
// which the CPU repeats in validate().
precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D u_tex;
uniform ivec2 u_origin;     // window position of the cell's lower left pixel
uniform vec3 u_scale;
uniform vec3 u_bias;
uniform bool u_gray;        // one-component formats: show .r as gray

out vec4 o_color;

void main()
{
    ivec2 t = (ivec2(gl_FragCoord.xy) - u_origin) / 8;
    uvec4 v = texelFetch(u_tex, t, 0);
    vec3 c = u_gray ? vec3(float(v.r)) : vec3(v.rgb);
    o_color = vec4(clamp(c * u_scale + u_bias, 0.0, 1.0), 1.0);
}
