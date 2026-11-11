#version 300 es
// Depth-only pass: the framebuffer has no color attachment and the fragment shader writes nothing.
// The depth written is the rasterized window z (ES 3.0 §2.13.1, §3.6.1). In the depth-stencil
// formats the stencil buffer receives the object id through glStencilOp(KEEP, KEEP, REPLACE).
void main()
{
}
