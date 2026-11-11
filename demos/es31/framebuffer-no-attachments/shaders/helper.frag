#version 310 es
// gl_HelperInvocation. The shader asks for derivatives (dFdx / dFdy of the window position), so on the
// edges of a primitive the 2 x 2 quads of pixels that are only partly covered need invocations for the
// uncovered pixels ("helper invocations", GLSL ES 3.10 section 7.1.2). A helper must not have side effects:
// every store and atomic below is performed by helpers as well, and must be without effect.
//   guarded     counts the invocations that run the atomic behind if (!gl_HelperInvocation)
//   unguarded   counts every invocation that executes the shader, with no guard at all
//   helper_seen is incremented only when gl_HelperInvocation is true (a helper's atomic would show here)
//   deriv_bad   counts real invocations whose derivative of gl_FragCoord is not (1, 1) within 0.01 (printed only:
//               section 8.13 leaves derivative accuracy open)
//   cells[]     per-pixel count of the unguarded atomic
precision highp float;
precision highp int;

layout(std430, binding = 0) buffer Counters
{
    uint guarded;
    uint unguarded;
    uint helper_seen;
    uint deriv_bad;
    uint cells[];
} c;

uniform int u_stride;

void main()
{
    float dx = dFdx(gl_FragCoord.x);
    float dy = dFdy(gl_FragCoord.y);
    if (abs(dx - 1.0) > 0.01 || abs(dy - 1.0) > 0.01) atomicAdd(c.deriv_bad, 1u);

    if (!gl_HelperInvocation) atomicAdd(c.guarded, 1u);
    atomicAdd(c.unguarded, 1u);
    if (gl_HelperInvocation) atomicAdd(c.helper_seen, 1u);

    uvec2 p = uvec2(gl_FragCoord.xy);
    uint index = p.y * uint(u_stride) + p.x;
    if (index < uint(c.cells.length())) atomicAdd(c.cells[index], 1u);
}
