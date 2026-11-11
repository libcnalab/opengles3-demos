#version 300 es
// Shows a visualization texture (RGBA8) over the current panel rectangle, "cover"-fitted: the
// texture keeps its aspect ratio and is cropped symmetrically so the panel is always filled.
precision highp float;
precision mediump sampler2D;

uniform sampler2D u_image;
uniform vec4 u_rect;   // x0, y0, width, height of the panel in window pixels
uniform vec4 u_uv;     // scale.xy, offset.xy: uv = offset + local * scale

out vec4 o_color;

void main()
{
    vec2 local = (gl_FragCoord.xy - u_rect.xy) / u_rect.zw;
    o_color = texture(u_image, u_uv.zw + local * u_uv.xy);
}
