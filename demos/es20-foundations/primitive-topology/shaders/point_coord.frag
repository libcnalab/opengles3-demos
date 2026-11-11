#version 300 es
// Visualizes gl_PointCoord: red = s (0 at the left edge of the point sprite, 1 at the right edge),
// green = t (0 at the TOP edge, 1 at the BOTTOM edge - t grows downwards, ES 3.0 equations 3.1/3.2).
//
// gl_PointCoord is only meaningful while rasterizing GL_POINTS; for lines and triangles its value
// is undefined, which is why this is a separate program that is used only for the point draw.
precision mediump float;

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = vec4(gl_PointCoord, 0.0, 1.0);
}
