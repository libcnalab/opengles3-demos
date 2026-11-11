#version 320 es
// One triangle that covers the viewport; no vertex buffer (gl_VertexID selects the corner).
precision highp float;

out vec2 v_ndc;

void main()
{
    vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);  // (-1,-1) (3,-1) (-1,3)
    v_ndc = p;
    gl_Position = vec4(p, 0.0, 1.0);
}
