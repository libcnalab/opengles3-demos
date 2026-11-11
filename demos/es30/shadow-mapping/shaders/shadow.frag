#version 300 es
// Pass 1 (depth only): the framebuffer has no color attachment (glDrawBuffers(NONE)), so the
// fragment shader declares no output. Depth is written by the fixed-function pipeline.
precision highp float;

void main()
{
}
