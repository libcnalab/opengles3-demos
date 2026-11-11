#version 300 es
// Draws one rectangle (a TRIANGLE_STRIP of 4 vertices) from a ring-buffer slot that the CPU rewrote for this
// update. Position is already in normalized device coordinates; the color is flat.
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec3 a_color;

out vec3 v_color;

void main()
{
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_color = a_color;
}
