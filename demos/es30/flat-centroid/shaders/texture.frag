#version 300 es
// Magnified (NEAREST) view of the resolved centroid comparison image.
precision highp float;

uniform sampler2D u_image;

in vec2 v_uv;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(texture(u_image, v_uv).rgb, 1.0);
}
