#version 300 es
// Program 1, vertex stage. The uniform block Palette holds everything the scene draws with; this stage
// uses only `offset` (the square's center), the fragment stage uses the rest, so the block is referenced
// by BOTH stages. The block is declared with an instance name (pal), so the members are written pal.offset
// in the shader, while the GL names them Palette.offset (block name, not instance name).
//
// std140 offsets, written out because the application fills a C++ struct instead of querying them:
//   color[3] 0 (array stride 16), offset 48, gain 56, pick 60, total 64 bytes.
layout(std140) uniform Palette
{
    vec4  color[3];
    vec2  offset;
    float gain;
    int   pick;
} pal;

void main()
{
    // A TRIANGLE_STRIP of four vertices forms a square: corners (-1,-1) (1,-1) (-1,1) (1,1).
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
    gl_Position = vec4(pal.offset + corner * 0.35, 0.0, 1.0);
}
