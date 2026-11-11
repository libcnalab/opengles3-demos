#version 300 es
// Reads a 3D texture (unit 0) or a 2D array texture (unit 1) in the way selected by u_mode:
//
//   0  texture(sampler3D, vec3(corner, u_r))        filtered lookup, r is the NORMALIZED depth coordinate
//   1  texture(sampler2DArray, vec3(corner, u_r))   filtered lookup, r is the LAYER coordinate (not normalized)
//   2  texelFetch(sampler3D, ivec3(i, j, u_index), u_lod)       one exact texel, no filtering, no wrapping
//   3  texelFetch(sampler2DArray, ivec3(i, j, u_index), u_lod)
//   4  textureSize(sampler3D, u_lod)       the size is written into the color: channel = size * 20 / 255
//   5  textureSize(sampler2DArray, u_lod)  (for arrays the third component is the number of layers)
//
// sampler3D and sampler2DArray have NO default precision in GLSL ES 3.00 (section 4.5.4: only sampler2D and
// samplerCube have one), so both must be declared. The two sampler uniforms point at different units: two
// samplers of different types must never refer to the same unit, or the draw call is INVALID_OPERATION.
precision highp float;
precision highp int;
precision highp sampler3D;
precision highp sampler2DArray;

uniform sampler3D u_tex3d;
uniform sampler2DArray u_tex_array;
uniform int u_mode;
uniform float u_r;
uniform int u_index;
uniform int u_lod;

in vec2 v_corner;
out vec4 o_color;

void main()
{
    vec4 c = vec4(0.0);
    if (u_mode == 0)
    {
        c = texture(u_tex3d, vec3(v_corner, u_r));
    }
    else if (u_mode == 1)
    {
        c = texture(u_tex_array, vec3(v_corner, u_r));
    }
    else if (u_mode == 2)
    {
        // The corner (0..1) is mapped onto the texels of level u_lod: the size comes from textureSize.
        ivec3 size = textureSize(u_tex3d, u_lod);
        ivec2 ij = min(ivec2(v_corner * vec2(size.xy)), size.xy - 1);
        c = texelFetch(u_tex3d, ivec3(ij, u_index), u_lod);
    }
    else if (u_mode == 3)
    {
        ivec3 size = textureSize(u_tex_array, u_lod);
        ivec2 ij = min(ivec2(v_corner * vec2(size.xy)), size.xy - 1);
        c = texelFetch(u_tex_array, ivec3(ij, u_index), u_lod);
    }
    else if (u_mode == 4)
    {
        c = vec4(vec3(textureSize(u_tex3d, u_lod)) * 20.0 / 255.0, 1.0);
    }
    else
    {
        c = vec4(vec3(textureSize(u_tex_array, u_lod)) * 20.0 / 255.0, 1.0);
    }
    o_color = c;
}
