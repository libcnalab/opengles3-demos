#version 320 es
// Vertex stage of the query-only program (never drawn): the program exists so that glGetnUniform*
// has uniforms of four different types to read back.
void main()
{
    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
