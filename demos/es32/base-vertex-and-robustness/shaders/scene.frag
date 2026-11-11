#version 320 es
// Every mesh has one colour on all four vertices, so the interpolated colour is constant across a quad.
precision highp float;

in highp vec3 v_color;
layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(v_color, 1.0);
}
