#version 310 es
// Like fs_vertical.frag but the split is horizontal: the BOTTOM half shows the forwarded tint, the
// top half this program's own u_bg. Replacing the fragment stage rotates the pattern by 90 degrees.
precision highp float;

layout(location = 1) uniform vec3 u_bg;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_tint;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(v_uv.y < 0.5 ? v_tint : u_bg, 1.0);
}
