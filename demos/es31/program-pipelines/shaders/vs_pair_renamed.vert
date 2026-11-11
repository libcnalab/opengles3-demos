#version 310 es
// Same locations and types as vs_pair_normal.vert but DIFFERENT NAMES. Because both sides carry a
// location qualifier, ES 3.2 section 7.4.1 matches by location ("the two variables are declared
// with the same location qualifier and match in type and qualification"), so this program forms a
// valid pipeline with the fragment programs whose inputs are called v_uv and v_tint.
layout(location = 0) uniform vec3 u_tint;

layout(location = 0) out vec2 uv_renamed;
layout(location = 1) out vec3 tint_renamed;

void main()
{
    vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    uv_renamed = corner;
    tint_renamed = u_tint;
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
