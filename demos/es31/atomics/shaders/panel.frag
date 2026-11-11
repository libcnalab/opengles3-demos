#version 310 es
// Draws a 128 x 128 mask (kept[], one word per pixel). When FRAGMENT_COUNTERS is defined (the application does so
// only if MAX_FRAGMENT_ATOMIC_COUNTER_BUFFERS and MAX_FRAGMENT_ATOMIC_COUNTERS allow it) the shader also counts, with
// atomic counters in the FRAGMENT stage, every fragment it shades and every lit one. Without multisampling a
// fullscreen triangle shades each pixel of the viewport exactly once (ES 3.2 section 7.11.1), so the counters are
// exactly 128 * 128 and the number of kept elements.
precision highp float;
precision highp int;

layout(std430, binding = 0) readonly buffer Kept { uint kept[]; } k;

#ifdef FRAGMENT_COUNTERS
layout(binding = 0, offset = 0) uniform atomic_uint c_fragments;
layout(binding = 0, offset = 4) uniform atomic_uint c_lit;
#endif

uniform float u_x0;       // window x of the left edge of the panel

out vec4 o_color;

void main()
{
    int x = int(gl_FragCoord.x - u_x0);
    int y = int(gl_FragCoord.y);
    bool lit = k.kept[y * 128 + x] != 0u;
#ifdef FRAGMENT_COUNTERS
    atomicCounterIncrement(c_fragments);
    if (lit) atomicCounterIncrement(c_lit);
#endif
    o_color = lit ? vec4(1.0, 0.8, 0.0, 1.0) : vec4(0.1, 0.1, 0.2, 1.0);
}
