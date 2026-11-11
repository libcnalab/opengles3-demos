#version 100
// @expect: link-pass
// @spec: GLSL ES 1.00 §8.7 Texture Lookup Functions - every 2D/cube form; "Lod" forms in the vertex shader, "bias" forms in the fragment shader
// @feature: GLSL100.TEXTURE2D
attribute vec4 a_position;
uniform sampler2D u_tex;
uniform samplerCube u_cube;
varying vec4 v_color;
void main()
{
    v_color = texture2D(u_tex, a_position.xy)
            + texture2DProj(u_tex, a_position.xyw) + texture2DProj(u_tex, a_position)
            + texture2DLod(u_tex, a_position.xy, 1.0)
            + texture2DProjLod(u_tex, a_position.xyw, 1.0) + texture2DProjLod(u_tex, a_position, 1.0)
            + textureCube(u_cube, a_position.xyz) + textureCubeLod(u_cube, a_position.xyz, 1.0);
    gl_Position = a_position;
}
