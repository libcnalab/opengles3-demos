#version 100
// texture2D with the fragment-only bias parameter (GLSL ES 1.00 §8.7): lambda = lambda_base + bias.
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D u_mip;
uniform float u_bias;
varying vec2 v_uv;

void main()
{
    gl_FragColor = texture2D(u_mip, v_uv, u_bias);
}
