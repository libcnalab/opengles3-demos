#version 300 es
// The interpolated cube position is the view direction; cube-map face selection does not need it
// normalized (ES 3.0 3.8.9). The cube map texels are already display (gamma-encoded) colours.
precision highp float;
precision highp samplerCube;  // samplers default to lowp in GLSL ES 3.00 (4.5.4)

uniform samplerCube u_sky;

in vec3 v_dir;
out vec4 o_color;

void main()
{
    o_color = vec4(texture(u_sky, v_dir).rgb, 1.0);
}
