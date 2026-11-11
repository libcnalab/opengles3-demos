#version 300 es
// Constant colour. Its alpha is what SAMPLE_ALPHA_TO_COVERAGE converts into a coverage mask.
precision highp float;

uniform vec4 u_color;

in vec2 v_uv;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_color;
}
