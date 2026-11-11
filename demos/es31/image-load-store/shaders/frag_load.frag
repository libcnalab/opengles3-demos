#version 310 es
// imageLoad in a fragment shader: shows an RGBA8 image color-inverted, one texel per pixel.
precision highp float;
layout(rgba8, binding = 0) uniform readonly highp image2D u_img;
uniform ivec2 u_origin;     // lower-left window pixel of the cell
layout(location = 0) out vec4 o_color;

void main()
{
    vec4 c = imageLoad(u_img, ivec2(gl_FragCoord.xy) - u_origin);
    o_color = vec4(1.0 - c.rgb, 1.0);
}
