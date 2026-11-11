#version 300 es
// Second pass: draws data that an earlier pass captured. Attribute locations 0, 2 and 3 are filled from
// the captured buffers (location 1 of the particle layout, the velocity, is not needed here).
layout(location = 0) in vec2 a_pos;
layout(location = 2) in vec3 a_color;
layout(location = 3) in float a_size;

out vec3 v_color;

void main()
{
    gl_Position = vec4(a_pos, 0.0, 1.0);
    gl_PointSize = a_size;  // ignored for triangles
    v_color = a_color;
}
