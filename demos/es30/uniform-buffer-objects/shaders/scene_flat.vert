#version 300 es
// Vertex stage of programs 2 and 3. It declares no uniform block at all, so a block that only the
// fragment stage uses is reported as NOT referenced by the vertex shader (ES 3.0 section 2.12.6.2).
// The square covers the cell (corners at +-0.9); v_pos goes from -1 to +1 across it.
out vec2 v_pos;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    v_pos = corner;
    gl_Position = vec4(corner * 0.9, 0.0, 1.0);
}
