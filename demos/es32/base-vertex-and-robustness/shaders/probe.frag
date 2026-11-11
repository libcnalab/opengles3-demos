#version 320 es
// Query-only program for glGetnUniformfv / iv / uiv. Every uniform contributes to the output so that
// all four stay active after linking (an inactive uniform has no location to query).
precision highp float;

uniform vec4 u_f;   // 4 floats  = 16 bytes  (glGetnUniformfv)
uniform ivec3 u_i;  // 3 ints    = 12 bytes  (glGetnUniformiv)
uniform uvec2 u_u;  // 2 uints   =  8 bytes  (glGetnUniformuiv)
uniform mat2 u_m;   // 4 floats  = 16 bytes, column-major (glGetnUniformfv)

layout(location = 0) out vec4 o_color;

void main()
{
    o_color = u_f + vec4(vec3(u_i), float(u_u.x + u_u.y)) + vec4(u_m[0], u_m[1]);
}
