#version 300 es
// The picture drawn into every texture image (cube faces, array layers, 3D slices): the base color
// u_base, dark at the left edge and brighter to the right, with horizontal stripes of 8 pixels.
// Every image gets its own base color, so a wrong face / layer / slice is visible and measurable.
precision highp float;
uniform vec3 u_base;
uniform float u_size;  // edge of the target image in pixels
out vec4 o_color;

void main()
{
    float shade = 0.35 + 0.65 * gl_FragCoord.x / u_size;
    float stripe = (((int(gl_FragCoord.y) >> 3) & 1) == 1) ? 0.8 : 1.0;
    o_color = vec4(u_base * shade * stripe, 1.0);
}
