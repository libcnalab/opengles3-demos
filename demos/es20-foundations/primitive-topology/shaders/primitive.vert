#version 300 es
// One vertex shader for every primitive type. The application draws points, lines and triangles
// from the same vertex buffer; this shader neither knows nor cares which one it is part of.

layout(location = 0) in vec2 a_position;    // window-independent clip-space x, y (w = 1)
layout(location = 1) in float a_point_size; // per-vertex point size in pixels

void main()
{
    gl_Position = vec4(a_position, 0.0, 1.0);

    // gl_PointSize is read only when the vertex belongs to a GL_POINTS primitive; for lines and
    // triangles the value is ignored. If a points draw leaves it unwritten the size is undefined
    // (ES 3.0 section 3.4), so the shader always writes it. The value is clamped to
    // ALIASED_POINT_SIZE_RANGE, whose upper bound is only guaranteed to be 1.
    gl_PointSize = a_point_size;
}
