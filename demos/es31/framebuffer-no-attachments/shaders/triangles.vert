#version 310 es
// Two triangles (6 vertices, GL_TRIANGLES) whose corners are passed in normalized device coordinates.
uniform vec2 u_pos[6];
uniform float u_depth;

void main()
{
    gl_Position = vec4(u_pos[gl_VertexID], u_depth * 2.0 - 1.0, 1.0);
}
