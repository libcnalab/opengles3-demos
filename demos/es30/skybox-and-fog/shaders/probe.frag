#version 300 es
// Samples a cube map along one fixed direction (a uniform), used to test seamless filtering exactly
// on a cube edge: the direction is not interpolated, so it is known exactly.
precision highp float;

uniform samplerCube u_cube;
uniform vec3 u_dir;

out vec4 o_color;

void main()
{
    o_color = vec4(texture(u_cube, u_dir).rgb, 1.0);
}
