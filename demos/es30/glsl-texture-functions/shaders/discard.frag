#version 300 es
// Fragment stage of the transform-feedback program. RASTERIZER_DISCARD is enabled while it runs, so it never executes;
// an ES 3.0 program object still needs a fragment shader to link.
precision mediump float;
out vec4 o_color;

void main()
{
    o_color = vec4(0.0);
}
