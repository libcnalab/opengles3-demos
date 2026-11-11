#version 300 es
// Reads a texture and writes it to the 8-bit default framebuffer. u_mode selects what is read:
//
//   0  texture(sampler2D, corner) * u_scale + u_bias            the filtered (here NEAREST) texel
//   1  textureLod(sampler2D, corner, u_lod_f) * u_scale + u_bias  an exactly chosen mip level
//   2  textureSize(sampler2D, u_lod)          the size is written into the color: channel = size * 20 / 255
//   3  textureSize(sampler3D, u_lod)
//   4  textureSize(sampler2DArray, u_lod)     (the third component of an array size is the layer count)
//
// The color buffer is 8-bit unsigned normalized, so values outside [0, 1] are clamped. The scale and bias bring
// textures with another range into it: snorm values [-1, 1] with 0.5 * v + 0.5, half-float values up to 4 with 0.25 * v.
//
// Samplers of different types sit on different texture units. sampler3D and sampler2DArray have no default
// precision (GLSL ES 3.00 section 4.5.4) and sampler2D defaults to lowp, which does not have to keep half-float or
// shared-exponent values: all three are declared highp.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
precision highp sampler2DArray;

uniform sampler2D u_tex2d;
uniform sampler3D u_tex3d;
uniform sampler2DArray u_tex_array;
uniform int u_mode;
uniform float u_lod_f;
uniform int u_lod;
uniform vec4 u_scale;
uniform vec4 u_bias;

in vec2 v_corner;
out vec4 o_color;

void main()
{
    vec4 c = vec4(0.0);
    if (u_mode == 0)
    {
        c = texture(u_tex2d, v_corner) * u_scale + u_bias;
    }
    else if (u_mode == 1)
    {
        c = textureLod(u_tex2d, v_corner, u_lod_f) * u_scale + u_bias;
    }
    else if (u_mode == 2)
    {
        c = vec4(vec2(textureSize(u_tex2d, u_lod)) * 20.0 / 255.0, 0.0, 1.0);
    }
    else if (u_mode == 3)
    {
        c = vec4(vec3(textureSize(u_tex3d, u_lod)) * 20.0 / 255.0, 1.0);
    }
    else
    {
        c = vec4(vec3(textureSize(u_tex_array, u_lod)) * 20.0 / 255.0, 1.0);
    }
    o_color = c;
}
