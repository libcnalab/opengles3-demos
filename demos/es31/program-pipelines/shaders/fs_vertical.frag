#version 310 es
// A separable FRAGMENT program: the left half of the viewport shows the tint that the vertex stage
// forwards, the right half shows this program's own u_bg uniform. u_bg sits at location 1 so that a
// program holding BOTH stages (progC in the demo) can keep u_tint (vertex, location 0) and u_bg
// (fragment, location 1) apart: in one program the two stages share one uniform name space.
precision highp float;   // highp inputs: identical precision to the vertex outputs (section 7.4.1)

layout(location = 1) uniform vec3 u_bg;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_tint;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(v_uv.x < 0.5 ? v_tint : u_bg, 1.0);
}
