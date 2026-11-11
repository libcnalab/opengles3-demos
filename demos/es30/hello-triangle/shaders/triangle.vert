#version 300 es
// GLSL ES 3.00: vertex inputs use `in` (ES 1.00's `attribute` is gone) and explicit
// locations replace glBindAttribLocation / glGetAttribLocation lookups.

layout(location = 0) in vec2 a_position;  // clip-space x, y
layout(location = 1) in vec3 a_color;

out vec3 v_color;  // interpolated across the triangle (default: smooth, perspective-correct)

void main()
{
    v_color = a_color;
    // w = 1, so clip coordinates equal normalized device coordinates.
    gl_Position = vec4(a_position, 0.0, 1.0);
}
