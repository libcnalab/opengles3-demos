#version 310 es
// The depth component of a DEPTH24_STENCIL8 / DEPTH32F_STENCIL8 texture (DEPTH_STENCIL_TEXTURE_MODE =
// DEPTH_COMPONENT, the initial value) through an ordinary sampler2D, shown as a gray level.
precision highp float;

layout(binding = 0) uniform highp sampler2D u_depth;

in vec2 v_uv;
out vec4 o_color;

void main()
{
    float d = texture(u_depth, v_uv).r;
    o_color = vec4(vec3(d), 1.0);
}
