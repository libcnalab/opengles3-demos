#version 320 es
// Cell D: an instanced geometry shader. layout(invocations = 6) runs the shader six times for every input
// triangle; gl_InvocationID selects the copy. Copy k is rotated about the Y axis by u_spin[k] = (cos, sin) and
// moved to u_place[k]. Output order (ES 3.2 section 11.3.4.2): all primitives of one input triangle, invocation 0
// first, before anything generated from the next input triangle.
precision highp float;

layout(triangles, invocations = 6) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) uniform mat4 u_viewproj;
layout(location = 1) uniform vec4 u_place[6];  // xyz = centre of copy k
layout(location = 7) uniform vec2 u_spin[6];   // (cos, sin) of the rotation of copy k

in VS_OUT
{
    vec3 world;
} gs_in[];

out GS_OUT
{
    vec3 world;
    flat vec3 normal;  // unnormalised face normal of this copy
    flat int inst;     // gl_InvocationID
    flat int face;     // gl_PrimitiveIDIn / 2: two triangles per cube face
} gs_out;

void main()
{
    int k = gl_InvocationID;
    vec2 cs = u_spin[k];
    vec3 q[3];
    for (int i = 0; i < 3; ++i)
    {
        vec3 p = gs_in[i].world;
        q[i] = u_place[k].xyz + vec3(cs.x * p.x + cs.y * p.z, p.y, -cs.y * p.x + cs.x * p.z);
    }
    vec3 n = cross(q[1] - q[0], q[2] - q[0]);
    for (int i = 0; i < 3; ++i)
    {
        gs_out.world = q[i];
        gs_out.normal = n;
        gs_out.inst = k;
        gs_out.face = gl_PrimitiveIDIn / 2;
        gl_Position = u_viewproj * vec4(q[i], 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
