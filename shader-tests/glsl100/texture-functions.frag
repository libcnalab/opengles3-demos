#version 100
precision mediump float;
uniform sampler2D u_tex;
uniform samplerCube u_cube;
varying vec4 v_color;
void main()
{
    vec4 p = v_color;
    gl_FragColor = texture2D(u_tex, p.xy) + texture2D(u_tex, p.xy, 0.5)
                 + texture2DProj(u_tex, p.xyw) + texture2DProj(u_tex, p.xyw, 0.5)
                 + texture2DProj(u_tex, p) + texture2DProj(u_tex, p, 0.5)
                 + textureCube(u_cube, p.xyz) + textureCube(u_cube, p.xyz, 0.5);
}
