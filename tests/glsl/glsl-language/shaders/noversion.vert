// Deliberately NO #version directive. GLSL ES 1.00 §3.4 and GLSL ES 3.00 §3.4: "shaders that do not
// include a #version directive will be treated as targeting version 1.00".
#if __VERSION__ != 100
#error "a shader without #version must be compiled as GLSL ES 1.00"
#endif
attribute vec2 a_position;
attribute vec2 a_uv;
uniform float u_q;
varying vec2 v_uv;
varying vec4 v_proj;

void main()
{
    v_uv = a_uv;
    // Projective texture coordinates (s*q, t*q, <ignored>, q): texture2DProj divides by the last component.
    v_proj = vec4(a_uv * u_q, 5.0, u_q);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
