#version 300 es
// Fills the attached image with a color that depends on the pixel position:
//     color = u_base + u_gx * x + u_gy * y        (x, y in (0,1): the pixel center divided by the image size)
// Different (base, gx, gy) triples make it possible to tell apart which slice or layer was rendered.
precision highp float;

uniform vec3 u_base;
uniform vec3 u_gx;
uniform vec3 u_gy;
uniform vec2 u_size;

out vec4 o_color;

void main()
{
    vec2 f = gl_FragCoord.xy / u_size;  // gl_FragCoord is the pixel center: (x + 0.5, y + 0.5)
    o_color = vec4(u_base + u_gx * f.x + u_gy * f.y, 1.0);
}
