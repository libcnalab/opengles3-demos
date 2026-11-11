#version 320 es
// Scene fragment shader. u_mode == 0 paints the flat colour u_color (the triangle), u_mode == 1 samples the
// 2x2 checker texture on unit 0 (the quad). Both uniforms are placed with layout(location) and the sampler
// with layout(binding), so the C++ side needs no glGetUniformLocation and no glUniform1i for the sampler.
precision mediump float;

layout(location = 0) uniform vec4 u_color;
layout(location = 1) uniform int u_mode;
layout(binding = 0) uniform mediump sampler2D u_tex;

in vec2 v_uv;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = (u_mode == 0) ? u_color : texture(u_tex, v_uv);
}
