#version 320 es
// One triangle that covers the whole viewport: (-1, -1), (3, -1), (-1, 3).
// Every sample of every pixel inside the viewport lies inside it, whatever the sample positions are,
// so the fill passes give every sample a value without depending on implementation-defined coverage.
void main()
{
    vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
