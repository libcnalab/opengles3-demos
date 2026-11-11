#version 100
// u_uv_scale is chosen by the application so that one pixel steps 1/16 of the texture in s and t:
// with an 8x8 base level that is rho = 0.5 texels per pixel, lambda_base = log2(0.5) = -1 exactly.
attribute vec2 a_position;
attribute vec2 a_uv;
uniform vec2 u_uv_scale;
varying vec2 v_uv;

void main()
{
    v_uv = a_uv * u_uv_scale;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
