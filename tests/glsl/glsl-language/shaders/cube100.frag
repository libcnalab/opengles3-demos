#version 100
// textureCube (GLSL ES 1.00 §8.7): the major axis of the direction selects the face.
precision mediump float;
uniform samplerCube u_cube;
uniform vec3 u_dir;

void main()
{
    gl_FragColor = textureCube(u_cube, u_dir);
}
