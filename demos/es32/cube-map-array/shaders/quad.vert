#version 320 es
// A quad covering the whole viewport (TRIANGLE_STRIP, 4 vertices, no vertex buffer). v_st are the face coordinates
// (0,0) at the top-left corner of the viewport with t growing downwards, like the rows of a texture upload.
out vec2 v_st;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_st = vec2(corner.x, 1.0 - corner.y);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
