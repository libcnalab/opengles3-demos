#version 300 es
// Axis-aligned quads given in pixels of the current viewport (coverage bands, zoom views, test quads).
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv;

uniform vec2 u_viewport_size;

out vec2 v_uv;

void main()
{
    v_uv = a_uv;
    gl_Position = vec4(a_position / u_viewport_size * 2.0 - 1.0, 0.0, 1.0);
}
