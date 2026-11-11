#version 300 es
// One interleaved vertex: a 2D position and a color. Every vertex is fetched by index, so the
// pixels show which vertices the index buffer selected - a wrong index type, offset or buffer
// binding picks other vertices and therefore other colors.
layout(location = 0) in vec2 a_position;  // NDC x, y
layout(location = 1) in vec3 a_color;

out vec3 v_color;

void main()
{
    v_color = a_color;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
