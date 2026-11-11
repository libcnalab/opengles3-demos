#version 310 es
// One instance per bar. The bar value is a per-instance integer attribute taken straight from a buffer that a compute
// shader wrote (the buffer is bound as ARRAY_BUFFER; glMemoryBarrier(VERTEX_ATTRIB_ARRAY_BARRIER_BIT) separates the
// shader writes from this fetch). The bar spans from the baseline to value * scale pixels above it (below, if negative).
layout(location = 0) in int a_value;

uniform ivec2 u_target;  // framebuffer size in pixels
uniform int u_x0;        // window x of the first bar
uniform int u_pitch;     // pixels between the starts of consecutive bars
uniform int u_width;     // bar width in pixels
uniform int u_baseline;  // window y of the value 0
uniform float u_scale;   // pixels per unit of value

void main()
{
    // Triangle strip: vertices 0, 1 on the baseline, 2, 3 at the tip.
    float x = float(u_x0 + gl_InstanceID * u_pitch + (gl_VertexID & 1) * u_width);
    float y = float(u_baseline) + float(gl_VertexID >> 1) * float(a_value) * u_scale;
    gl_Position = vec4(2.0 * vec2(x, y) / vec2(u_target) - 1.0, 0.0, 1.0);
}
