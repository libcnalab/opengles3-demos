#version 300 es
// Vertex shader of the "separate vertex format" demo. It knows nothing about buffers, bindings, strides or
// divisors: whatever the vertex array object delivers arrives in these three inputs.
//
//   location 0  a_pos     position (per vertex, or per instance if its binding has a divisor)
//   location 1  a_color   color; a normalized UNSIGNED_BYTE attribute arrives as 0..1
//   location 2  a_offset  added to the scaled position; when the array is disabled the current value
//                         (0, 0, 0, 1) is used, so the offset is zero
//
// Where each value comes from (which buffer, which byte offset, which stride, how often it advances) is set
// with glVertexAttribFormat / glVertexAttribBinding / glBindVertexBuffer / glVertexBindingDivisor
// (ES 3.2 section 10.3.1, 10.3.2, 10.3.7).
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec4 a_color;
layout(location = 2) in vec2 a_offset;

uniform vec2 u_scale;

out vec4 v_color;

void main()
{
    gl_Position = vec4(a_pos * u_scale + a_offset, 0.0, 1.0);
    v_color = a_color;
}
