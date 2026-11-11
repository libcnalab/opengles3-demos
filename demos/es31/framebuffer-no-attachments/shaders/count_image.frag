#version 310 es
// The image variant of count.frag: the side effects are imageStore calls on two r32ui images (GLSL ES 3.10 has
// imageLoad / imageStore / imageSize; image atomics only arrive with GLSL ES 3.20, so no counting here).
//   u_marker   (unit 0)  (y << 16) | x, the invocation's own window position
//   u_covered  (unit 1)  1
// Overlapping triangles store the same values to the same pixel, so the order of the stores does not matter.
// The framebuffer being drawn into has no attachment; the images are bound with glBindImageTexture.
precision highp float;
precision highp int;

layout(r32ui, binding = 0) writeonly uniform highp uimage2D u_marker;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D u_covered;

void main()
{
    ivec2 p = ivec2(gl_FragCoord.xy);
    imageStore(u_marker, p, uvec4((uint(p.y) << 16) | uint(p.x), 0u, 0u, 0u));
    imageStore(u_covered, p, uvec4(1u, 0u, 0u, 0u));
}
