#version 300 es
// One oversized triangle that covers the whole viewport, generated from gl_VertexID: the corners are
// (-1,-1), (3,-1) and (-1,3) in clip space, so no vertex buffer or attribute is needed and there is no
// diagonal seam. Every post-processing pass is a draw of these three vertices with a fragment shader
// that reads its inputs with texelFetch at ivec2(gl_FragCoord.xy).
void main()
{
    vec2 corner = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
    gl_Position = vec4(corner, 0.0, 1.0);
}
