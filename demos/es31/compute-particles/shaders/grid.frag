#version 310 es
precision highp float;

in vec2 v_xz;
in float v_major;
uniform vec3 u_background;
layout(location = 0) out vec4 o_color;

void main()
{
    // Lines fade with the distance from the fountain so the grid gives depth without framing the picture.
    float d = length(v_xz);
    float k = exp(-0.22 * d) * mix(0.35, 0.75, v_major);
    o_color = vec4(mix(u_background, vec3(0.30, 0.42, 0.62), k), 1.0);
}
