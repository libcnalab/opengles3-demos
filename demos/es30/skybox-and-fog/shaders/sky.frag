#version 300 es
// The interpolated cube position is the direction from the eye through this fragment. It does not
// need to be normalized: cube map face selection only compares the magnitudes of its components and
// divides by the largest one (ES 3.0 section 3.8.9, Table 3.21).
precision highp float;

uniform samplerCube u_sky;

in vec3 v_dir;
out vec4 o_color;

void main()
{
    o_color = vec4(texture(u_sky, v_dir).rgb, 1.0);
}
