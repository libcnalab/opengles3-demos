#version 310 es
// INTERFACE MISMATCH (location). The names are identical to the fragment program's inputs, but the
// location qualifiers differ (3 and 4 instead of 0 and 1). With locations present, matching is by
// location only (ES 3.2 section 7.4.1; GLSL ES 3.10 section 9.2.1 footnote 2), so the fragment
// inputs at locations 0 and 1 have no matching output: a program pipeline made of this program and a
// fragment program with v_uv/v_tint at 0/1 fails validation (ES 3.2 section 11.1.3.11).
layout(location = 0) uniform vec3 u_tint;

layout(location = 3) out vec2 v_uv;
layout(location = 4) out vec3 v_tint;

void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = corner;
    v_tint = u_tint;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
