#version 310 es
// INTERFACE MISMATCH (type). Same locations as the fragment program, but the output at location 0
// is a vec3 while the fragment input at location 0 is a vec2: "match in type" fails (ES 3.2
// section 7.4.1), so the pipeline is invalid.
layout(location = 0) uniform vec3 u_tint;

layout(location = 0) out vec3 v_uv;
layout(location = 1) out vec3 v_tint;

void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = vec3(corner, 0.0);
    v_tint = u_tint;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
