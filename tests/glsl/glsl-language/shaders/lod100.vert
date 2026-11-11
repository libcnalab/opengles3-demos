#version 100
// Texture lookups in the VERTEX shader (GLSL ES 1.00 §8.7): the "Lod" built-ins are vertex-only and use
// lod directly as the level of detail. The texture has one solid colour per mipmap level.
attribute vec2 a_position;
uniform sampler2D u_mip;
uniform samplerCube u_cube;
uniform int u_mode;
uniform float u_lod;
varying vec4 v_color;

void main()
{
    if (u_mode == 0)
        v_color = texture2DLod(u_mip, vec2(0.3, 0.6), u_lod);
    else if (u_mode == 1)
        v_color = texture2DProjLod(u_mip, vec3(0.6, 1.2, 2.0), u_lod);
    else if (u_mode == 2)
        v_color = texture2DProjLod(u_mip, vec4(0.6, 1.2, 9.0, 2.0), u_lod);
    else
        v_color = textureCubeLod(u_cube, vec3(0.1, -0.2, -1.0), u_lod);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
