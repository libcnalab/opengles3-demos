#version 300 es
precision highp float;

uniform vec4 u_front_color;
uniform vec4 u_back_color;

layout(location = 0) out vec4 o_color;

void main()
{
    // gl_FrontFacing (GLSL ES 3.00 section 7.2) is true for fragments of front-facing triangles,
    // where "front" follows glFrontFace and the signed window-space area (ES 3.0 section 3.6.1).
    // For fragments of points and lines it is always true: "all other fragments are considered
    // front-facing" (ES 3.0 section 3.9.2.2). The back color is only observable when back faces
    // are rasterized at all, i.e. when culling does not remove them.
    o_color = gl_FrontFacing ? u_front_color : u_back_color;
}
