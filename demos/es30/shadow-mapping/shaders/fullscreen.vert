#version 300 es
// One oversized triangle that covers the whole viewport; no vertex buffer, only gl_VertexID.
// Used for the shadow-map inset and for the depth read-back pass.
out vec2 v_uv;

void main()
{
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));  // (0,0) (2,0) (0,2)
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
