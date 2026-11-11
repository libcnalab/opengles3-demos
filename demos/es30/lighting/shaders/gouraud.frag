#version 300 es
precision highp float;

in vec3 v_color;  // lit colour computed per vertex and interpolated (perspective-correct)

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(pow(clamp(v_color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
