#version 300 es
// A ground plane in perspective: a TRIANGLE_STRIP of 4 vertices generated from gl_VertexID. Texture coordinates run
// 0..96 across (REPEAT) and 0..1 into the distance, so the far end is strongly minified and texture() walks down the
// mip chain of the 32x16 texture.
uniform mat4 u_mvp;
out vec2 v_uv;

void main()
{
    vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
    v_uv = vec2(corner.x * 96.0, corner.y);
    vec3 world = vec3((corner.x - 0.5) * 48.0, 0.0, 4.0 - corner.y * 90.0);
    gl_Position = u_mvp * vec4(world, 1.0);
}
