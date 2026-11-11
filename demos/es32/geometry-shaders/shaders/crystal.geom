#version 320 es
// Cell B: every input TRIANGLE becomes a prism ("crystal"): a lifted cap (one 3-vertex strip) and three side
// quads (one 4-vertex strip each), separated by EndPrimitive(): 3 + 3 * 4 = 15 vertices per invocation.
// The lift is the unnormalised face normal (twice the triangle area) times a factor chosen by
// gl_PrimitiveIDIn % 3, so large faces grow taller and neighbouring faces differ. Only exact operations are used
// (subtract, cross product, multiply by 0.5/1/1.5, add) so that transform feedback can be compared bit for bit.
precision highp float;

layout(triangles) in;
layout(triangle_strip, max_vertices = 15) out;

layout(location = 0) uniform mat4 u_viewproj;
layout(location = 1) uniform float u_lift;

in VS_OUT
{
    vec3 world;
} gs_in[];

out GS_OUT
{
    vec3 world;
    flat vec3 normal;  // unnormalised outward face normal of the cap or of the side quad
    flat int prim;     // gl_PrimitiveIDIn, captured by transform feedback
} gs_out;

void emit_vertex(vec3 p, vec3 n)
{
    gs_out.world = p;
    gs_out.normal = n;
    gs_out.prim = gl_PrimitiveIDIn;
    gl_Position = u_viewproj * vec4(p, 1.0);
    gl_PrimitiveID = gl_PrimitiveIDIn;  // read by the fragment shader (11.3.4.4: every vertex of the primitive)
    EmitVertex();
}

void main()
{
    vec3 p[3] = vec3[3](gs_in[0].world, gs_in[1].world, gs_in[2].world);
    vec3 n = cross(p[1] - p[0], p[2] - p[0]);
    vec3 m = n * (u_lift * float(1 + gl_PrimitiveIDIn % 3));

    emit_vertex(p[0] + m, n);
    emit_vertex(p[1] + m, n);
    emit_vertex(p[2] + m, n);
    EndPrimitive();

    for (int k = 0; k < 3; ++k)
    {
        vec3 a = p[k];
        vec3 b = p[(k + 1) % 3];
        vec3 side = cross(b - a, n);  // outward: the triangle is counter-clockwise
        emit_vertex(a, side);
        emit_vertex(a + m, side);
        emit_vertex(b, side);
        emit_vertex(b + m, side);
        EndPrimitive();
    }
}
